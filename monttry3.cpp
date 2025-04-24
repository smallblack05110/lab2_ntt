#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <tuple>
using namespace std;

class MontMul
{
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
    return REDC(a * R2);
  }

  // 从 Montgomery 域转换回普通整数
  uint64_t fromMont(uint64_t aR) const
  {
    return REDC(aR);
  }

  // 在 Montgomery 域内进行乘法运算
  uint64_t mulMont(uint64_t aR, uint64_t bR) const
  {
    return REDC(aR * bR);
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
};

void fRead(int *a, int *b, int *n, int *p, int input_id)
{
  string str1 = "/nttdata/";
  string str2 = to_string(input_id);
  string strin = str1 + str2 + ".in";
  char data_path[strin.size() + 1];
  copy(strin.begin(), strin.end(), data_path);
  data_path[strin.size()] = '\0';
  ifstream fin;
  fin.open(data_path, ios::in);
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

void fWrite(int *ab, int n, int input_id)
{
  string str1 = "files/";
  string str2 = to_string(input_id);
  string strout = str1 + str2 + ".out";
  char output_path[strout.size() + 1];
  copy(strout.begin(), strout.end(), output_path);
  output_path[strout.size()] = '\0';
  ofstream fout;
  fout.open(output_path, ios::out);
  for (int i = 0; i < n * 2 - 1; ++i)
  {
    fout << ab[i] << '\n';
  }
  fout.close();
}

void fCheck(int *ab, int n, int input_id)
{
  string str1 = "/nttdata/";
  string str2 = to_string(input_id);
  string strout = str1 + str2 + ".out";
  char data_path[strout.size() + 1];
  copy(strout.begin(), strout.end(), data_path);
  data_path[strout.size()] = '\0';
  ifstream fin;
  fin.open(data_path, ios::in);
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

int quick_mod(int a, int b, int p)
{ // 快速计算a的b次方
  int result = 1;
  a = a % p;
  while (b > 0)
  {
    if (b % 2 == 1)
    {
      result = (1LL * result * a) % p; // 奇数就多乘一个a
    }
    a = (1LL * a * a) % p; // 底数自乘
    b /= 2;
  }
  return result;
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

void ntt_iter(vector<long long> &a, int p, int root, bool invert, const MontMul &mont)
{
  int n = a.size();
  for (int i = 1, j = 0; i < n; ++i)
  {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j |= bit;
    if (i < j)
      swap(a[i], a[j]);
  }

  for (int len = 2; len <= n; len <<= 1)
  {
    int wn = quick_mod(root, (p - 1) / len, p);
    if (invert)
      wn = quick_mod(wn, p - 2, p);
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

vector<long long> get_result(vector<long long> &a, vector<long long> &b, int p, int root, const MontMul &mont)
{
  int n = a.size();
  ntt_iter(a, p, root, false, mont);
  ntt_iter(b, p, root, false, mont);
  vector<long long> c(n);
  for (int i = 0; i < n; ++i)
    c[i] = mont.mulMont(a[i], b[i]);
  ntt_iter(c, p, root, true, mont);
  int inv_n = quick_mod(n, p - 2, p);
  long long invR = mont.toMont(inv_n);
  for (int i = 0; i < n; ++i)
    c[i] = mont.mulMont(c[i], invR);
  return c;
}

int a[300000], b[300000], ab[300000];

int main(int argc, char *argv[])
{
  int test_begin = 0, test_end = 1;
  for (int id = test_begin; id <= test_end; ++id)
  {
    long double ans = 0;
    int n_, p_;
    fRead(a, b, &n_, &p_, id);

    int len = 1;
    while (len < 2 * n_)
      len <<= 1;
    fill(a + n_, a + len, 0);
    fill(b + n_, b + len, 0);

    vector<long long> va(a, a + len), vb(b, b + len);
    long long R = 1LL << 30;
    MontMul mont(R, p_);
    for (int i = 0; i < len; ++i)
    {
      va[i] = mont.toMont(va[i]);
      vb[i] = mont.toMont(vb[i]);
    }

    int root = 3;
    auto start = chrono::high_resolution_clock::now();
    vector<long long> cr = get_result(va, vb, p_, root, mont);
    auto end = chrono::high_resolution_clock::now();
    ans = chrono::duration<double, ratio<1, 1000>>(end - start).count();

    for (int i = 0; i < 2 * n_ - 1; ++i)
    {
      ab[i] = (int)mont.fromMont(cr[i]);
    }

    fCheck(ab, n_, id);
    cout << "average latency for n = " << n_ << " p = " << p_ << " : " << ans << " (us)" << endl;
    fWrite(ab, n_, id);
  }
  return 0;
}