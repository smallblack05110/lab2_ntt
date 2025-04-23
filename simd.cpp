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
#include <arm_neon.h> // 使用 NEON 加速
using namespace std;

// 扩展欧几里得算法求解 a*x + b*y = gcd(a, b)，返回 (g, x, y)
static std::tuple<long long, long long, long long> egcd(long long a, long long b)
{
  if (a == 0)
    return std::make_tuple(b, 0LL, 1LL);
  auto result = egcd(b % a, a);
  long long g = std::get<0>(result);
  long long x = std::get<1>(result);
  long long y = std::get<2>(result);
  // 更新 x, y 对应原始 a, b
  return std::make_tuple(g, y - (b / a) * x, x);
}

class MontMul
{
public:
  MontMul(long long R, long long N)
  {
    this->N = N;
    this->R = R;
    this->logR = static_cast<int>(log2(R));
    // 计算 N_inv，使得 N * N_inv ≡ 1 (mod R)
    long long N_inv = std::get<1>(egcd(N, R));
    if (N_inv < 0)
    {
      N_inv += R;
    }
    this->N_inv_neg = R - N_inv;
    // R2 = R * R mod N
    long long R_modN = R % N;
    this->R2 = (R_modN * R_modN) % N;
  }

  // 蒙哥马利约减：计算 (T + m * N) / R mod N，其中 m = (T * N_inv_neg) mod R
  long long REDC(long long T) const
  {
    // mask = R - 1
    long long mask = (logR == 64 ? -1LL : ((1LL << logR) - 1));
    long long m = ((T & mask) * N_inv_neg) & mask;
    long long t = (T + m * N) >> logR;
    if (t >= N)
    {
      t -= N;
    }
    return t;
  }

  // 转换到蒙哥马利表述（a -> a * R mod N）
  long long toMont(long long a) const
  {
    // 乘以 R^2 后简化，得到 a 的蒙哥马利表示
    long long T = (long long)((__int128)a * R2 % N);
    return REDC(T);
  }

  // 从蒙哥马利表示还原普通值
  long long fromMont(long long aR) const
  {
    return REDC(aR);
  }

  // 蒙哥马利模乘：相乘两个蒙哥马利表示数
  long long mulMont(long long aR, long long bR) const
  {
    long long T = (long long)((__int128)aR * bR);
    return REDC(T);
  }

  // 批量转换 vector 到蒙哥马利域
  void toMontVec(vector<long long> &inout) const
  {
#pragma omp parallel for
    for (int i = 0; i < (int)inout.size(); ++i)
    {
      __int128 prod = (__int128)inout[i] * R2;
      long long T = (long long)(prod % N);
      inout[i] = REDC(T);
    }
  }

  // 批量从蒙哥马利域转换 vector
  void fromMontVec(vector<long long> &inout) const
  {
#pragma omp parallel for
    for (int i = 0; i < (int)inout.size(); ++i)
    {
      inout[i] = REDC(inout[i]);
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
  long long N;
  long long R;
  long long N_inv_neg;
  long long R2;
  int logR;
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
  int test_begin = 0, test_end = 1;
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
    long long R = 1LL << 30;
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