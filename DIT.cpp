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

// 优化 NTT 支持任意 p
inline long long fpow(long long a, long long b, int p)
{
  long long res = 1;
  a %= p;
  for (; b; b >>= 1)
  {
    if (b & 1)
      (res *= a) %= p;
    (a *= a) %= p;
  }
  return res;
}
inline void butterfly(int f[], int l)
{
  static int tr[300000], last;
  if (last != l)
  {
    last = l;
    for (int i = 1; i < 1 << l; ++i)
      tr[i] = tr[i >> 1] >> 1 | (i & 1) * (1 << l - 1);
  }
  for (int i = 1; i < 1 << l; ++i)
    if (tr[i] < i)
      swap(f[tr[i]], f[i]);
}
inline void reverse(int f[], int l, int p)
{
  const long long invl = fpow(1 << l, p - 2, p);
  for (int i = 0; i < 1 << l; ++i)
    f[i] = invl * f[i] % p;
  std::reverse(f + 1, f + (1 << l));
}
template <bool rev>
inline void DIT(int f[], int l, int p, int gen)
{
  butterfly(f, l);
  for (int len = 2, j = 0; len <= 1 << l; len <<= 1, ++j)
  {
    const long long w_n = fpow(gen, (p - 1) / len, p);
    long long g, h, w = 1;
    for (int st = 0; st < 1 << l; st += len, w = 1)
      for (int i = st; i < st + len / 2; ++i, (w *= w_n) %= p)
        g = f[i], h = f[i + len / 2] * w % p,
        f[i] = (g + h) % p,
        f[i + len / 2] = (p + g - h) % p;
  }
  if (rev)
    reverse(f, l, p);
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

    vector<int> va(a, a + len), vb(b, b + len);
    auto start = chrono::high_resolution_clock::now();
    DIT<false>(va.data(), log2(len), p_, 3); // 3 是常用原根
    DIT<false>(vb.data(), log2(len), p_, 3);
    for (int i = 0; i < len; ++i)
        va[i] = 1LL * va[i] * vb[i] % p_;
    DIT<true>(va.data(), log2(len), p_, 3);
    auto end = chrono::high_resolution_clock::now();
    ans = chrono::duration<double, std::ratio<1, 1000>>(end - start).count();
    for (int i = 0; i < 2 * n_ - 1; ++i)
        ab[i] = va[i];

    // 验证结果正确性并输出耗时
    fCheck(ab, n_, id);
    cout << "average latency for n = " << n_ << " p = " << p_ << " : " << ans << " (us)" << endl;
    fWrite(ab, n_, id);
  }
  return 0;
}