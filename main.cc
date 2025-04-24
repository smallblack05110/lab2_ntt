#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <tuple>
#include <arm_neon.h>
#include <stdexcept>
#include <cstdint>
#include <omp.h>
using namespace std;

class MontMul
{
public:
  // 构造函数要求 R 为 2 的幂
  MontMul(uint64_t R, uint64_t N) : R(R), N(N)
  {
    if (R == 0 || (R & (R - 1)) != 0)
    {
      throw std::invalid_argument("R must be a power of two");
    }
    logR = static_cast<int>(std::log2(R));
    if ((1ULL << logR) != R)
    {
      throw std::invalid_argument("R is not a power of two");
    }
    uint64_t N_inv = modinv(N, R);
    N_inv_neg = R - N_inv;
    __int128 R_squared = static_cast<__int128>(R) * R;
    R2 = static_cast<uint64_t>(R_squared % N);
  }

  // REDC 算法，将 __int128 类型的 T 转换为 Montgomery 域内元素
  uint64_t REDC(__int128 T) const
  {
    uint64_t mask = (logR == 64) ? ~0ULL : ((1ULL << logR) - 1);
    uint64_t m_part = static_cast<uint64_t>(T) & mask;
    uint64_t m = (m_part * N_inv_neg) & mask;
    __int128 mN = static_cast<__int128>(m) * N;
    __int128 t_val = (T + mN) >> logR;
    uint64_t t = static_cast<uint64_t>(t_val);
    return t >= N ? t - N : t;
  }

  // 将普通整数转换到 Montgomery 域
  uint64_t toMont(uint64_t a) const
  {
    return REDC(static_cast<__int128>(a) * R2);
  }

  // 从 Montgomery 域转换回普通整数
  uint64_t fromMont(uint64_t aR) const
  {
    return REDC(aR);
  }

  // 在 Montgomery 域内进行乘法运算
  uint64_t mulMont(uint64_t aR, uint64_t bR) const
  {
    return REDC(static_cast<__int128>(aR) * bR);
  }

  // 保持原有接口：对于 a, b（要求均小于模 N），返回 a * b mod N
  uint64_t ModMul(uint64_t a, uint64_t b)
  {
    if (a >= N || b >= N)
    {
      throw std::invalid_argument("input must be smaller than modulus N");
    }
    uint64_t aR = toMont(a);
    uint64_t bR = toMont(b);
    uint64_t abR = mulMont(aR, bR);
    return fromMont(abR);
  }

  // 批量将 vector 中的元素转换到 Montgomery 域
  void toMontVec(vector<long long> &inout) const
  {
#pragma omp parallel for
    for (size_t i = 0; i < inout.size(); ++i)
    {
      uint64_t val = static_cast<uint64_t>(inout[i]);
      inout[i] = static_cast<long long>(toMont(val));
    }
  }

  // 批量将 Montgomery 域内的元素转换回普通整数
  void fromMontVec(vector<long long> &inout) const
  {
#pragma omp parallel for
    for (size_t i = 0; i < inout.size(); ++i)
    {
      uint64_t val = static_cast<uint64_t>(inout[i]);
      inout[i] = static_cast<long long>(fromMont(val));
    }
  }

  void mulMontVec(const vector<long long> &a, const vector<long long> &b, vector<long long> &out) const
  {
    int len = a.size();
    const int64x2_t vN = vdupq_n_s64(N);
    const int64x2_t vMask = vdupq_n_s64((1LL << logR) - 1);
    const int64x2_t vNinv = vdupq_n_s64(N_inv_neg);

    for (int i = 0; i < len; i += 2)
    {
      // 加载两个64位整数到向量寄存器
      int64x2_t va = vld1q_s64(reinterpret_cast<const int64_t *>(&a[i]));
      int64x2_t vb = vld1q_s64(reinterpret_cast<const int64_t *>(&b[i]));
      // 由于没有 vmulq_s64，我们分别计算每个 64 位乘法
      int64_t a0 = vgetq_lane_s64(va, 0);
      int64_t a1 = vgetq_lane_s64(va, 1);
      int64_t b0 = vgetq_lane_s64(vb, 0);
      int64_t b1 = vgetq_lane_s64(vb, 1);
      // 计算乘积（使用 __int128 确保不会溢出）
      int64_t T0 = (long long)((__int128)a0 * b0);
      int64_t T1 = (long long)((__int128)a1 * b1);
      // 重新组合成向量
      int64x2_t vT = vsetq_lane_s64(T0, vT, 0);
      vT = vsetq_lane_s64(T1, vT, 1);
      // 计算 m = (T & mask) * N_inv_neg & mask
      int64x2_t vT_low = vandq_s64(vT, vMask);
      int64_t m0 = (long long)((__int128)vgetq_lane_s64(vT_low, 0) * N_inv_neg) & ((1LL << logR) - 1);
      int64_t m1 = (long long)((__int128)vgetq_lane_s64(vT_low, 1) * N_inv_neg) & ((1LL << logR) - 1);
      int64x2_t vm = vsetq_lane_s64(m0, vm, 0);
      vm = vsetq_lane_s64(m1, vm, 1);
      // 计算 T + m * N
      int64_t tmp0 = T0 + m0 * N;
      int64_t tmp1 = T1 + m1 * N;
      int64x2_t vTmp = vsetq_lane_s64(tmp0, vTmp, 0);
      vTmp = vsetq_lane_s64(tmp1, vTmp, 1);

      // 右移 logR 位
      int64x2_t vt = vshrq_n_s64(vTmp, logR);

      // 检查是否 >= N，如果是则减去 N
      uint64x2_t cmp = vcgeq_s64(vt, vN);
      int64x2_t vCond = vsubq_s64(vt, vN);
      vt = vbslq_s64(cmp, vCond, vt);

      // 存储结果
      vst1q_s64(reinterpret_cast<int64_t *>(&out[i]), vt);
    }
  }

private:
  uint64_t N;
  uint64_t R;
  int logR;
  uint64_t N_inv_neg;
  uint64_t R2;

  struct EgcdResult
  {
    int64_t g;
    int64_t x;
    int64_t y;
  };

  // 扩展欧几里得算法（适用于无符号数，但 x, y 为带符号类型）
  static EgcdResult egcd(uint64_t a, uint64_t b)
  {
    uint64_t old_r = a, r = b;
    int64_t old_s = 1, s = 0;
    int64_t old_t = 0, t = 1;
    while (r != 0)
    {
      uint64_t quotient = old_r / r;
      uint64_t temp = old_r;
      old_r = r;
      r = temp - quotient * r;

      int64_t temp_s = old_s;
      old_s = s;
      s = temp_s - static_cast<int64_t>(quotient) * s;

      int64_t temp_t = old_t;
      old_t = t;
      t = temp_t - static_cast<int64_t>(quotient) * t;
    }
    return {static_cast<int64_t>(old_r), old_s, old_t};
  }

  static uint64_t modinv(uint64_t a, uint64_t m)
  {
    auto result = egcd(a, m);
    if (result.g != 1)
    {
      throw std::runtime_error("modular inverse does not exist");
    }
    int64_t x = result.x % static_cast<int64_t>(m);
    if (x < 0)
    {
      x += m;
    }
    return static_cast<uint64_t>(x);
  }
};

// 快速幂取模
int quick_mod(int a, int b, int p)
{
  long long result = 1;
  long long base = a % p;
  while (b > 0)
  {
    if (b % 2 == 1)
    {
      result = (result * base) % p;
    }
    base = (base * base) % p;
    b /= 2;
  }
  return (int)result;
}

// void ntt_recur(vector<int> &a, int p, int root, bool invert, MontMul &mont)
// { // ntt递归实现
//   int n = a.size();
//   if (n == 1) // 等于一时直接返回
//     return;

//   int half = n / 2;
//   vector<int> a_e(half), a_o(half);
//   for (int i = 0; i < half; ++i)
//   {
//     a_e[i] = a[2 * i];     // 偶数项
//     a_o[i] = a[2 * i + 1]; // 奇数项
//   }
//   ntt_recur(a_e, p, root, invert, mont);
//   ntt_recur(a_o, p, root, invert, mont);

//   int wn = quick_mod(root, (p - 1) / n, p);
//   if (invert)
//   {
//     wn = quick_mod(wn, p - 2, p); // 如果是反变换，wn要取模p-2（费马小定理）
//   }

//   int w0 = 1;
//   for (int i = 0; i < half; ++i)
//   {
//     int op1 = a_e[i];
//     int op2 = mont.ModMul(w0, a_o[i]); // 使用蒙哥马利模乘
//     a[i] = (op1 + op2) % p;
//     a[i + half] = (op1 - op2 + p) % p;
//     w0 = mont.ModMul(w0, wn); // 使用蒙哥马利模乘
//   }
// }

// 迭代实现 NTT (Number Theoretic Transform)
void ntt_iter(vector<long long> &a, int p, int root, bool invert, const MontMul &mont)
{
  int n = a.size();
  // 位逆序重排
  for (int i = 1, j = 0; i < n; ++i)
  {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1)
    {
      j ^= bit;
    }
    j |= bit;
    if (i < j)
    {
      swap(a[i], a[j]);
    }
  }
  // 按长度迭代合并
  for (int len = 2; len <= n; len <<= 1)
  {
    int wn = quick_mod(root, (p - 1) / len, p);
    if (invert)
    {
      wn = quick_mod(wn, p - 2, p); // 若为逆变换，则使用 wn 的逆元
    }
    long long wnR = mont.toMont(wn);
    for (int i = 0; i < n; i += len)
    {
      long long w = mont.toMont(1);
      for (int j = 0; j < len / 2; ++j)
      {
        long long u = a[i + j];
        long long v = mont.mulMont(w, a[i + j + len / 2]);
        a[i + j] = (u + v) % p;
        a[i + j + len / 2] = (u - v + p) % p;
        w = mont.mulMont(w, wnR);
      }
    }
  }
}

// 逐点相乘（蒙哥马利域）并执行 NTT 计算卷积，返回蒙哥马利域下的结果
vector<long long> get_result(vector<long long> &a, vector<long long> &b, int p, int root, const MontMul &mont)
{
  int n = a.size();
  ntt_iter(a, p, root, false, mont);
  ntt_iter(b, p, root, false, mont);
  vector<long long> c(n);
  // 逐点相乘（蒙哥马利域）
  mont.mulMontVec(a, b, c);
  // 逆 NTT
  ntt_iter(c, p, root, true, mont);
  // 乘以 n 的逆元 (mod p)
  int inv_n = quick_mod(n, p - 2, p);
  vector<long long> invVec(n, mont.toMont(inv_n));
  mont.mulMontVec(c, invVec, c);
  return c;
}

void fRead(int *a, int *b, int *n, int *p, int input_id)
{
  // 数据输入函数
  string str1 = "/nttdata/";
  string str2 = to_string(input_id);
  string strin = str1 + str2 + ".in";
  char data_path[256];
  strncpy(data_path, strin.c_str(), sizeof(data_path));
  data_path[sizeof(data_path) - 1] = '\0';
  ifstream fin;
  fin.open(data_path, ios::in);
  if (!fin)
  {
    cerr << "无法打开输入文件: " << strin << endl;
    return;
  }
  fin >> *n >> *p;
  for (int i = 0; i < *n; ++i)
  {
    fin >> a[i];
  }
  for (int i = 0; i < *n; ++i)
  {
    fin >> b[i];
  }
  fin.close();
}

void fCheck(int *ab, int n, int input_id)
{
  // 判断多项式乘法结果是否正确
  string str1 = "/nttdata/";
  string str2 = to_string(input_id);
  string strout = str1 + str2 + ".out";
  char data_path[256];
  strncpy(data_path, strout.c_str(), sizeof(data_path));
  data_path[sizeof(data_path) - 1] = '\0';
  ifstream fin;
  fin.open(data_path, ios::in);
  if (!fin)
  {
    cerr << "无法打开输出文件: " << strout << endl;
    return;
  }
  for (int i = 0; i < n * 2 - 1; ++i)
  {
    int x;
    fin >> x;
    if (x != ab[i])
    {
      cout << "多项式乘法结果错误" << endl;
      fin.close();
      return;
    }
  }
  cout << "多项式乘法结果正确" << endl;
  fin.close();
}

void fWrite(int *ab, int n, int input_id)
{
  // 数据输出函数, 可以用来输出最终结果, 也可用于调试时输出中间数组
  string str1 = "files/";
  string str2 = to_string(input_id);
  string strout = str1 + str2 + ".out";
  char output_path[256];
  strncpy(output_path, strout.c_str(), sizeof(output_path));
  output_path[sizeof(output_path) - 1] = '\0';
  ofstream fout;
  fout.open(output_path, ios::out);
  if (!fout)
  {
    cerr << "无法打开输出文件用于写入: " << strout << endl;
    return;
  }
  for (int i = 0; i < n * 2 - 1; ++i)
  {
    fout << ab[i] << '\n';
  }
  fout.close();
}

// 全局输入和输出数组
int a[300000], b[300000], ab[300000];
int main(int argc, char *argv[])
{
  // 保证输入的所有模数的原根均为 3, 且模数都能表示为 a \times 4 ^ k + 1 的形式
  // 输入模数分别为 7340033 104857601 469762049 263882790666241
  // 第四个模数超过了整型表示范围, 如果实现此模数意义下的多项式乘法需要修改框架
  // 对第四个模数的输入数据不做必要要求, 如果要自行探索大模数 NTT, 请在完成前三个模数的基础代码及优化后实现大模数 NTT
  // 输入文件共五个, 第一个输入文件 n = 4, 其余四个文件分别对应四个模数, n = 131072
  // 在实现快速数论变化前, 后四个测试样例运行时间较久, 推荐调试正确性时只使用输入文件 1
  int test_begin = 0, test_end = 3;
  for (int id = test_begin; id <= test_end; ++id)
  {
    long double ans = 0.0;
    int n_, p_;
    // 读取输入的多项式和模数
    fRead(a, b, &n_, &p_, id);
    memset(ab, 0, sizeof(ab));
    // 计算卷积所需的长度（2 的幂）
    int len = 1;
    while (len < 2 * n_)
    {
      len <<= 1;
    }
    // 拓展多项式 a, b 使其长度达到 2n
    fill(a + n_, a + len, 0);
    fill(b + n_, b + len, 0);

    // 拷贝到 vector<long long>，用于 NTT（蒙哥马利域运算）
    vector<long long> va(a, a + len), vb(b, b + len);
    uint64_t R = 1ULL << 30;
    MontMul mont(R, p_);

    // 将系数转换到蒙哥马利域
    mont.toMontVec(va);
    mont.toMontVec(vb);

    int root = 3; // NTT 原根
    auto start = chrono::high_resolution_clock::now();
    // 使用 NTT 在蒙哥马利域执行卷积
    vector<long long> cr = get_result(va, vb, p_, root, mont);
    auto end = chrono::high_resolution_clock::now();
    ans = chrono::duration<double, std::ratio<1, 1000>>(end - start).count();

    // 将结果从蒙哥马利域转换回普通域
    mont.fromMontVec(cr);
    // 将结果存储到整数输出数组
    for (int i = 0; i < 2 * n_ - 1; ++i)
    {
      ab[i] = (int)cr[i];
    }

    // 验证结果正确性并输出耗时
    fCheck(ab, n_, id);
    cout << "average latency for n = " << n_ << " p = " << p_ << " : " << ans << " (us)" << endl;
    fWrite(ab, n_, id);
  }
  return 0;
}
