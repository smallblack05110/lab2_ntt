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
using namespace std;

void fRead(int *a, int *b, int *n, int *p, int input_id) {
    string str1 = "/nttdata/";
    string str2 = to_string(input_id);
    string strin = str1 + str2 + ".in";
    char data_path[strin.size() + 1];
    copy(strin.begin(), strin.end(), data_path);
    data_path[strin.size()] = '\0';
    ifstream fin;
    fin.open(data_path, ios::in);
    fin >> *n >> *p;
    for (int i = 0; i < *n; ++i) {
        fin >> a[i];
    }
    for (int i = 0; i < *n; ++i) {   
        fin >> b[i];
    }
    fin.close();
}

void fCheck(int *ab, int n, int input_id) {
    string str1 = "/nttdata/";
    string str2 = to_string(input_id);
    string strout = str1 + str2 + ".out";
    char data_path[strout.size() + 1];
    copy(strout.begin(), strout.end(), data_path);
    data_path[strout.size()] = '\0';
    ifstream fin;
    fin.open(data_path, ios::in);
    for (int i = 0; i < n * 2 - 1; ++i) {
        int x;
        fin >> x;
        if (x != ab[i]) {
            cout << "多项式乘法结果错误" << endl;
            fin.close();
            return;
        }
    }
    cout << "多项式乘法结果正确" << endl;
    fin.close();
}

void fWrite(int *ab, int n, int input_id) {
    string str1 = "files/";
    string str2 = to_string(input_id);
    string strout = str1 + str2 + ".out";
    char output_path[strout.size() + 1];
    copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';
    ofstream fout;
    fout.open(output_path, ios::out);
    for (int i = 0; i < n * 2 - 1; ++i) {
        fout << ab[i] << '\n';
    }
    fout.close();
}
int quick_mod(int a, int b, int p) {  //快速计算a的b次方      
    int result = 1;
    a =  a % p;
    while (b > 0) {
        if (b % 2 == 1) {
            result = (1LL * result * a) % p;    //奇数就多乘一个a
        }
        a = (1LL * a * a) % p;  //底数自乘
        b /= 2;
    }
    return result;
}

void ntt_recur(vector<int> &a, int p, int root, bool invert) {  //ntt递归实现
    int n = a.size();
    if (n == 1)     //等于一时直接返回
    return; 

    int half = n / 2;
    vector<int> a_e(half), a_o(half);
    for (int i = 0; i < half; ++i) {
        a_e[i] = a[2 * i];   //偶数项
        a_o[i] = a[2 * i + 1];   //奇数项
    }
    ntt_recur(a_e, p, root, invert);
    ntt_recur(a_o, p, root, invert);

    int wn = quick_mod(root, (p-1)/n, p);
    if (invert) {
        wn = quick_mod(wn, p-2, p);   //如果是反变换，wn要取模p-2（费马小定理）
    }

    int w0 = 1; 
    for (int i = 0; i < half; ++i) {
        int op1 = a_e[i];
        int op2 = (1LL * a_o[i] * w0) % p;
        a[i] = (op1 + op2) % p;
        a[i + half] = (op1 - op2 + p) % p;
        w0 = (1LL * w0 * wn) % p;
    }
}

void ntt_iter(vector<int>& a, int p, int root, bool invert) {  //ntt迭代实现
    int n = a.size();
    int half = n / 2;
    for (int i = 1, j = 0; i < n; i++) { 
         int bit = half;
        for (; j >= bit; bit /= 2) {
            j -= bit;
        }
        j += bit;
        if (i < j) {
            swap(a[i], a[j]);
        }
    }
    
    for (int len = 2; len <= n; len *= 2) {
        int wn;
        wn = quick_mod(3, (p - 1) / len, p);
        if(invert) {
            wn = quick_mod(wn, p - 2, p);
        }
        for (int i = 0; i < n; i += len) {
            int w0 = 1;
            for (int j = 0; j < len / 2; j++) {
                int op1 = a[i + j];
                int op2 = (1LL * a[i + j + len / 2] * w0) % p;
                a[i + j] = (op1 + op2) % p;
                a[i + j + len / 2] = (op1 - op2 + p) % p;
                w0 = (1LL * w0 * wn) % p;
            }
        }
    }
}

vector<int> get_result(vector<int> &a_vec, vector<int> &b_vec, int p, int len, int root) {
    ntt_recur(a_vec, p, root, false);
    ntt_recur(b_vec, p, root, false);
    vector<int> c_vec(len);
    for (int j = 0; j < len; ++j) {
        c_vec[j] = (1LL * a_vec[j] * b_vec[j]) % p;
    }
    ntt_recur(c_vec, p, root, true);
    int inv_m = quick_mod(len, p - 2, p);
    for (int j = 0; j < len; ++j) {
        c_vec[j] = (1LL * c_vec[j] * inv_m) % p;    //每项再除以len（模p下的乘法逆元）
    }
    return c_vec;
}

int a[300000], b[300000], ab[300000];

int main(int argc, char *argv[]) {
    int test_begin = 0;
    int test_end = 1;
    for (int i = test_begin; i <= test_end; ++i) {
        long double ans = 0;
        int n_, p_;
        fRead(a, b, &n_, &p_, i);

        int len = 1;
        while (len < 2 * n_) {
            len <<= 1;
        }

        fill(a + n_, a + len, 0);   //拓展多项式a，b 使其长度达到2n
        fill(b + n_, b + len, 0);
        vector<int> a_vec(a, a + len);
        vector<int> b_vec(b, b + len);
        int root = 3;
        auto Start = chrono::high_resolution_clock::now();
        vector<int> c_vec = get_result(a_vec, b_vec, p_, len, root);
        for (int j = 0; j < 2 * n_ - 1; ++j) {
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
