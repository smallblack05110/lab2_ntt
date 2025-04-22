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
#include <cstdint>
#include <stdexcept>
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
      uint64_t tmp_r = old_r;
      old_r = r;
      r = tmp_r - quotient * r;

      int64_t tmp_s = old_s;
      old_s = s;
      s = tmp_s - static_cast<int64_t>(quotient) * s;

      int64_t tmp_t = old_t;
      old_t = t;
      t = tmp_t - static_cast<int64_t>(quotient) * t;
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
    else
    {
      int64_t x = result.x % static_cast<int64_t>(m);
      if (x < 0)
      {
        x += m;
      }
      return static_cast<uint64_t>(x);
    }
  }

public:
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

  uint64_t REDC(__int128 T)
  {
    uint64_t mask = (logR == 64) ? ~0ULL : (1ULL << logR) - 1;
    uint64_t m_part = static_cast<uint64_t>(T) & mask;
    uint64_t m = (m_part * N_inv_neg) & mask;

    __int128 mN = static_cast<__int128>(m) * N;
    __int128 t_val = (T + mN) >> logR;
    uint64_t t = static_cast<uint64_t>(t_val);

    return t >= N ? t - N : t;
  }

  uint64_t ModMul(uint64_t a, uint64_t b)
  {
    if (a >= N || b >= N)
    {
      throw std::invalid_argument("input must be smaller than modulus N");
    }

    __int128 aR2 = static_cast<__int128>(a) * R2;
    uint64_t aR = REDC(aR2);

    __int128 bR2 = static_cast<__int128>(b) * R2;
    uint64_t bR = REDC(bR2);

    __int128 T = static_cast<__int128>(aR) * bR;
    uint64_t abR = REDC(T);

    return REDC(abR);
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

void ntt_recur(vector<int> &a, int p, int root, bool invert, MontMul &mont)
{ // ntt递归实现
  int n = a.size();
  if (n == 1) // 等于一时直接返回
    return;

  int half = n / 2;
  vector<int> a_e(half), a_o(half);
  for (int i = 0; i < half; ++i)
  {
    a_e[i] = a[2 * i];     // 偶数项
    a_o[i] = a[2 * i + 1]; // 奇数项
  }
  ntt_recur(a_e, p, root, invert, mont);
  ntt_recur(a_o, p, root, invert, mont);

  int wn = quick_mod(root, (p - 1) / n, p);
  if (invert)
  {
    wn = quick_mod(wn, p - 2, p); // 如果是反变换，wn要取模p-2（费马小定理）
  }

  int w0 = 1;
  for (int i = 0; i < half; ++i)
  {
    int op1 = a_e[i];
    int op2 = mont.ModMul(w0, a_o[i]); // 使用蒙哥马利模乘
    a[i] = (op1 + op2) % p;
    a[i + half] = (op1 - op2 + p) % p;
    w0 = mont.ModMul(w0, wn); // 使用蒙哥马利模乘
  }
}

void ntt_iter(vector<int> &a, int p, int root, bool invert, MontMul &mont)
{ // ntt迭代实现
  int n = a.size();
  int half = n / 2;
  int j = 0;
  for (int i = 1; i < n; i++)
  {
    int bit = n >> 1; // 每次都重新初始化
    for (; j & bit; bit >>= 1)
    {
      j ^= bit;
    }
    j |= bit;
    if (i < j)
      std::swap(a[i], a[j]);
  }

  for (int len = 2; len <= n; len *= 2)
  {
    int wn;
    wn = quick_mod(root, (p - 1) / len, p);
    if (invert)
    {
      wn = quick_mod(wn, p - 2, p);
    }
    for (int i = 0; i < n; i += len)
    {
      int w0 = 1;
      for (int j = 0; j < len / 2; j++)
      {
        int op1 = a[i + j];
        int op2 = mont.ModMul(w0, a[i + j + len / 2]); // 使用蒙哥马利模乘
        a[i + j] = (op1 + op2) % p;
        a[i + j + len / 2] = (op1 - op2 + p) % p;
        w0 = mont.ModMul(w0, wn); // 使用蒙哥马利模乘
      }
    }
  }
}

vector<int> get_result(vector<int> &a_vec, vector<int> &b_vec, int p, int len, int root, MontMul &mont)
{
  ntt_iter(a_vec, p, root, false, mont);
  ntt_iter(b_vec, p, root, false, mont);
  vector<int> c_vec(len);
  for (int j = 0; j < len; ++j)
  {
    c_vec[j] = mont.ModMul(a_vec[j], b_vec[j]); // 使用蒙哥马利模乘
  }
  ntt_iter(c_vec, p, root, true, mont);
  int inv_m = quick_mod(len, p - 2, p);
  for (int j = 0; j < len; ++j)
  {
    c_vec[j] = mont.ModMul(c_vec[j], inv_m); // 使用蒙哥马利模乘
  }
  return c_vec;
}

int a[300000], b[300000], ab[300000];

int main(int argc, char *argv[])
{
  int test_begin = 0;
  int test_end = 1;
  for (int i = test_begin; i <= test_end; ++i)
  {
    long double ans = 0;
    int n_, p_;
    fRead(a, b, &n_, &p_, i);

    int len = 1;
    while (len < 2 * n_)
    {
      len <<= 1;
    }

    fill(a + n_, a + len, 0); // 拓展多项式a，b 使其长度达到2n
    fill(b + n_, b + len, 0);
    vector<int> a_vec(a, a + len);
    vector<int> b_vec(b, b + len);

    int root = 3;

    // 初始化MontMul类
    long long R = 1LL << 30; // 选择R为2^30
    MontMul mont(R, p_);

    auto Start = chrono::high_resolution_clock::now();
    vector<int> c_vec = get_result(a_vec, b_vec, p_, len, root, mont);
    for (int j = 0; j < 2 * n_ - 1; ++j)
    {
      ab[j] = c_vec[j];
    }

    auto End = chrono::high_resolution_clock::now();
    chrono::duration<double, ratio<1, 1000>> elapsed = End - Start;
    ans += elapsed.count();

    fCheck(ab, n_, i);
    cout << "average latency for n = " << n_ << " p = " << p_ << " : " << ans << " (us) " << endl;
    fWrite(ab, n_, i);
  }
  return 0;
}
