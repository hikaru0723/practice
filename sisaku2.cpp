#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <cmath>
#include <numeric>   // std::accumulate に必要
#include <algorithm> // std::max に必要
#include <chrono>
#include <sstream>
#include <iomanip> // setprecisionを使用するのに必要
// aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

const double PI = 3.14159265358979323846;

using namespace std;

using Matrix = vector<vector<double>>;

//===========================================================================================
// 二項係数
double Comb(double& n, double& k)
{
    if (k < 0 || k > n) return 0.0;

    double res = 1.0;

    for (int i = 1; i <= k; ++i)
    {
        res *= (n - k + i);
        res /= i;
    }

    return res;
}
//========================================================================================

//=======================================================================================
// s階差分行列を生成
Matrix generate_Ds(double& n, double& s)
{
    Matrix Ds(n-s, vector<double>(n, 0.0));
    for(int i = 0; i < n-s; i++){
        for(int j = i; j < s+i+1; j++){
            double k = j - i;
            Ds[i][j] = pow(-1, s-j+i) * Comb(s, k);
        }
    }
    return Ds;
}

// Matrix generate_Ds(double& n, double& s, double& norm2)
// {
//     Matrix Ds(n-s, vector<double>(n, 0.0));

//     vector<double> c(s + 1);

//     for(int k = 0; k <= s; k++){
//         double kk = k;
//         c[k] = pow(-1, s - kk) * Comb(s, kk);
//         norm2 += c[k] * c[k];
//     }

//     double norm = sqrt(norm2);

//     for(int i = 0; i < n-s; i++){
//         for(int k = 0; k <= s; k++){
//             Ds[i][i + k] = c[k] / norm;
//         }
//     }

//     return Ds;
// }
//=========================================================================================

//=======================================================================================
//s階差分行列同士の内積 D_s^T D_s を生成
Matrix generate_DsDs(Matrix& Ds, double& n, double& s)
{
    Matrix DsDs(n, vector<double>(n, 0.0));
    int rows = n - s;

    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            for(int k = 0; k < rows; ++k){
                DsDs[i][j] += Ds[k][i] * Ds[k][j];
            }
        }
    }

    return DsDs;
}
//=========================================================================================

//==============================================================================================
// 表示
void printMatrix(const Matrix& A)
{
    int rows = A.size();
    int colms = A[0].size();
    for (int i = 0; i < rows; ++i){
        for (int j = 0; j < colms; ++j){
            cout << A[i][j] << "\t";
        }
        cout << endl;
    }
}
//==============================================================================================


//=================================================================================================================
//
std::vector<double> pro_WS(
    std::vector<double>& inp, 
    std::vector<double>& true_signal, 
    std::vector<double>& phi,
    Matrix& Ds,
    Matrix& DsDs,
    double& n, 
    double& lambda, 
    double& b, 
    double& sigma2, 
    double& mse,
    double& like,
    double& s,
    double& diff,
    double& ips, 
    int& gs_iter
)
{
    //===========================================================================================
    //中心化処理
    double ave = 0.0;
    for(int i=0; i < n; i++) ave += inp[i];
    ave /= n;
    for(int i=0; i < n; i++) inp[i] -= ave;
    //===========================================================================================

    //推定信号の初期化
    std::vector<double> est(n);
    est = inp;

    //ハイパーパラメータの設定
    double lambda_lr = 1e-3;
    double b_lr = 1e-7;
    // int gs_iter = 0;

    // Ds^T*Ds の対角成分
    std::vector<double> den_D(n);
    for(int i = 0; i < n; ++i) den_D[i] = DsDs[i][i];

    // std::vector<double> phi(n);
    // for (int i = 0; i < n; ++i){
    //     double sin = std::sin(i * PI / (2.0 * n));
    //     // phi[i] = std::pow(4.0, s) * std::pow(sin, 2.0*s);
    //     phi[i] = std::pow(4.0, s) * std::pow(sin, 2.0*s) / norm2;
    // }

    std::vector<double> psi(n);
    std::vector<double> kai(n);
    std::vector<double> den_Q(n); 
    std::vector<double> est_old(n);
    
    // || m_new - m_old || < eps まで反復
    while(1){
        gs_iter++;
        est_old = est;
        for (int i = 0; i < n; ++i) den_Q[i] = 1.0 / sigma2 + b + den_D[i] * lambda;

        // ガウスザイデルの対角成分
        // vector<double> den(n);

        // for (int i = 0; i < n; ++i)den[i] = 1.0 / sigma2 + lambda * DsDs[i][i];

        for (int t = 0; t < 2; ++t){
            for (int i = 0; i < n; ++i){
                double sum = 0.0;
                for (int j = 0; j < n; ++j){
                    if (j == i) continue;
                    sum += DsDs[i][j] * est[j];
                }
                est[i] =(inp[i] / sigma2 - lambda * sum)/ den_Q[i];
            }
        }
        
        //最尤推定に必要な値の計算
        for (int i = 0; i < n; ++i) {
            psi[i] = lambda * phi[i] + b;
            kai[i] = psi[i] + 1.0 / sigma2;
        }

        double sum_phi_psi_kai = 0.0;
        for (int i = 0; i < n; ++i)sum_phi_psi_kai += phi[i] / (psi[i] * kai[i]);

        double sum_psi_kai = 0.0;
        for (int i = 0; i < n; ++i)sum_psi_kai += 1.0 / (psi[i] * kai[i]);

        double sum_kai = 0.0;
        for (int i = 0; i < n; ++i)sum_kai += 1.0 / kai[i];

        //==========================================================================================================
        // mLambda
        vector<double> mLambda(n, 0.0);
        vector<double> c(s + 1);
        for (int i = 0; i <= s; ++i){
            double k = i;
            c[i] = pow(-1.0, s - k) * Comb(s, k);
        }

        // double norm2 = 0.0;

        // for (int i = 0; i <= s; ++i){
        //     double k = i;
        //     c[i] = pow(-1.0, s - k) * Comb(s, k);
        //     norm2 += c[i] * c[i];
        // }

        // double norm = sqrt(norm2);

        // for (int i = 0; i <= s; ++i){
        //     c[i] /= norm;
        // }
        //==========================================================================================================


        int rows = n - s;
        for (int r = 0; r < rows; ++r){
            // r 行目の差分
            double diff = 0.0;
            for (int k = 0; k <= s; ++k) diff += c[k] * est[r + k];
            // 転置側へ戻す
            for (int k = 0; k <= s; ++k)mLambda[r + k] += c[k] * diff;
        }

        double sum_mLambdam = 0.0;
        for (int i = 0; i < n; ++i) sum_mLambdam += mLambda[i] * est[i];

        double sum_m2 = 0.0;
        for (int i = 0; i < n; ++i) sum_m2 += est[i] * est[i];

        double sum_ym2 = 0.0;
        for (int i = 0; i < n; ++i) {
            double diff = inp[i] - est[i];
            sum_ym2 += diff * diff;
        }

        double grad_lambda = -sum_mLambdam / (2*n) + sum_phi_psi_kai / (2*n*sigma2);
        double grad_b = -sum_m2 / (2*n) + sum_psi_kai / (2*n*sigma2);

        // パラメータ更新
        lambda += lambda_lr * grad_lambda;
        // b += b_lr * grad_b;
        sigma2 = (sum_kai + sum_ym2) / n;

        //評価指標
        double diff = 0.0;
        for (int i = 0; i < n; i++) diff += std::abs(est[i] - est_old[i]);

        double sum_Dx2 = 0.0;
        for(int i = 0; i < n-1; i++) sum_Dx2 += (est[i+1] - est[i]) * (est[i+1] - est[i]);

        double sum_x2 = 0.0;
        for(int i = 0; i < n; i++) sum_x2 += est[i] * est[i];

        diff /= n;
        if (diff < ips) {        
            // std::cout << "Iter " << gs_iter 
            // << " lambda = " << lambda
            // << " b = " << b 
            // << " sigma2 = " << sigma2  
            // << " sum_ym2 = " << sum_ym2 
            // << " sum_kai = " << sum_kai 
            // << " diff = " << diff
            // << " sum_Dx2 = " << sum_Dx2
            // << " sum_x2 = " << sum_x2
            // << std::endl;

            break;
        }
    }

    //====================================================================================================================================
    // 尤度計算
    double mQm = 0.0;
    double sum_ln_kai = 0.0;
    double sum_ln_psi = 0.0;

    //--------------------------------------------
    // mLambda = D_s^T D_s est
    //--------------------------------------------
    vector<double> mLambda(n, 0.0);
    vector<double> c(s + 1);
    for (int i = 0; i <= s; ++i){
        double k = i;
        c[i] = pow(-1.0, s - k) * Comb(s, k);
    }

    // vector<double> c(s + 1);
    // double norm2 = 0.0;

    // for (int i = 0; i <= s; ++i){
    //     double k = i;
    //     c[i] = pow(-1.0, s - k) * Comb(s, k);
    //     norm2 += c[i] * c[i];
    // }
    
    // double norm = sqrt(norm2);

    // for (int i = 0; i <= s; ++i){
    //     c[i] /= norm;
    // }

    int rows = n - s;
    // D_s est → D_s^T(D_s est)
    for (int r = 0; r < rows; ++r){
        double diff = 0.0;
        // diff = (D_s est)_r
        for (int k = 0; k <= s; ++k) diff += c[k] * est[r + k];
        // mLambda += D_s^T diff
        for (int k = 0; k <= s; ++k) mLambda[r + k] += c[k] * diff;
    }

    //--------------------------------------------
    // mQm = est^T Q est
    //--------------------------------------------
    for (int i = 0; i < n; ++i) mQm += est[i] * (lambda * mLambda[i] + (b + 1.0 / sigma2) * est[i]);

    // for(int i=1; i<n-1; i++) sum_ln_kai += log(kai[i]);
    // for(int i=1; i<n-1; i++) sum_ln_psi += log(psi[i]);

    for(int i=0; i<n; i++) sum_ln_kai += log(kai[i]);
    for(int i=0; i<n; i++) sum_ln_psi += log(psi[i]);

    double sum_y2 = 0.0;
    for (int i = 0; i < n; ++i) sum_y2 += inp[i] * inp[i];

    // double like = 0.0;
    // like = 1/(2*n)*mQm - 1/(2*n)*sum_ln_kai + 1/(2*n)*sum_ln_psi - 1/2*log(2*PI*sigma2) - 1/(2*n*sigma2)*sum_y2;

    like = (1.0 / (2.0 * n)) * mQm
     - (1.0 / (2.0 * n)) * sum_ln_kai
     + (1.0 / (2.0 * n)) * sum_ln_psi
     - 0.5 * log(2.0 * PI * sigma2)
     - (1.0 / (2.0 * n * sigma2)) * sum_y2;
    //==============================================================================================================================

    for(int i=0; i < n; i++) est[i] += ave;
    for(int i=0; i < n; i++) inp[i] += ave;

    mse=0.0;
    for(int i=0.0; i < n; i++) mse += (true_signal[i] - est[i]) * (true_signal[i] - est[i]);
    mse /= n;

    // for(int i=0; i < n; i++) est[i] += ave;
    // for(int i=0; i < n; i++) inp[i] += ave;

    return est;
}

int main()
{
    // 現在時刻を取得
    auto start = std::chrono::high_resolution_clock::now();

    //======================================================================================
    //真の信号生成
    double n = 300.0;
    std::vector<double> x(n);
    // double step = 300.0 / n;
    double step = 300 / n;
    for (int i = 0; i < n; i++) x[i] = i * step;

    std::vector<double> true_signal(n);

    // for (int i = 0; i < n; i++){
    //     true_signal[i] 
    //     = std::exp(-(x[i] - 50) * (x[i] - 50) / 10) ;
    // }

    for (int i = 0; i < n; i++) {
        true_signal[i]
        = 0.3*std::sin((2*PI*x[i])/120) 
        + 0.2*std::sin((2*PI*x[i])/35) 
        + 0.15*std::sin((2*PI*x[i])/18);
    }

    // for (int i = 0; i < n; i++) {
    // true_signal[i]
    //     = 0.55 * std::sin(1.7 * x[i])
    //     + 0.28 * std::sin(5.3 * x[i] + 0.8)
    //     + 0.18 * std::sin(11.0 * x[i] - 0.4)
    //     + 0.08 * std::sin(23.0 * x[i]);
    // }

    // for (int i = 0; i < n; i++) {
    //     true_signal[i]
    //     = std::sin(x[i]);
    // }

    // for (int i = 0; i < n; i++) {
    //     true_signal[i] 
    //     = 0.75 * std::exp(-(x[i] - 10.0) * (x[i] - 10.0) / 5.0) 
    //     + std::exp(-(x[i] - 40.0) * (x[i] - 40.0) / 5.0) 
    //     + 0.5 * std::exp(-(x[i] - 70.0) * (x[i] - 70.0) / 5.0);
    // }

    // for (int i = 0; i < n; i++) {
    //     true_signal[i] 
    //     = 15 * std::exp(-(x[i] - 30) * (x[i] - 30) / 50) 
    //     + 20 * std::exp(-(x[i] - 120) * (x[i] - 120) / 60);
    // }

    // for (int i = 0; i < n; i++) {
    //     true_signal[i] 
    //     = 15 * std::exp(-(x[i] - 30) * (x[i] - 30) / 50) 
    //     + 20 * std::exp(-(x[i] - 120) * (x[i] - 120) / 60) 
    //     + 10 * std::exp(-(x[i] - 200) * (x[i] - 200) / 70);
    // }

    // for (int i = 0; i < n; i++) {
    //     true_signal[i] 
    //     = 15 * std::exp(-(x[i] - 30) * (x[i] - 30) / 10) 
    //     + 20 * std::exp(-(x[i] - 120) * (x[i] - 120) / 15) 
    //     + 10 * std::exp(-(x[i] - 200) * (x[i] - 200) / 40);
    // }
    //===================================================================================================================

    //=========================================================

    //=========================================================
    // 劣化信号
    double dev = 0.05;
    //======================================
    // 初期値設定
    double sigma2 = 0.1; 
    double lambda = 0.0;
    double lambda_i = 0.0;
    double s; // 次数を選択
    double b = 0.001;
    //======================================

    std::cout << "Please enter the rank of the difference matrix : ";
    std::cin >> s;

    std::cout << "Please enter the initial value for λ : ";
    std::cin >> lambda_i;

    //======================================
    // 劣化信号の作成
    int seed = 42;
    std::default_random_engine gen(seed);
    std::normal_distribution<> d(0, dev);
    std::vector<double> y_noisy(n);
    for(int i = 0; i < n; ++i) y_noisy[i] = true_signal[i] + d(gen);
    //======================================

    std::vector<double> est(n);
    std::vector<double> like_max_est(n);
    std::vector<double> mse_min_est(n);
    std::vector<double> est_first(n);
    double mse=0.0;
    double like=-10000.0;
    double like_max=-10000.0;
    double like_max_lambda=0.0;
    double dif_lambda = 10000.0;
    double best_mse_lambda=0.0;
    double mse_old = 10000.0;
    double mse_min_mse = 0.0;
    double like_max_mse = 0.0;
    double true_noisy_mse = 0.0;

    std::ofstream MSE("mse.csv");
    MSE << "lambda_i,lambda,mse,like,dif_lambda,b\n";

    // A = D_s^T D_s
    // Matrix A = generateDsTDs(n, s);

    // double norm2 = 0.0;

    Matrix Ds = generate_Ds(n, s);
    Matrix DsDs = generate_DsDs(Ds, n, s);

    std::vector<double> phi(n);
    for (int i = 0; i < n; ++i){
        double sin = std::sin(i * PI / (2.0 * n));
        phi[i] = std::pow(4.0, s) * std::pow(sin, 2.0*s);
        // phi[i] = std::pow(4.0, s) * std::pow(sin, 2.0*s) / norm2;
    }

    int iter;
    double add;

    std::cout << "Please enter the number of additions : ";
    std::cin >> iter;

    std::cout << "Please enter the value to add : ";
    std::cin >> add;

    std::cout << "--- Bayesian Optimization Start ---" << std::endl;

    for(int i = 0; i < iter; i++){
        double lambda = lambda_i;
        double like_old = like;
        double old_like_max_lambda = like_max_lambda;
        // double lambda_old = lambda;
        double dif_lambda2_min = dif_lambda * dif_lambda;
        double diff = 0.0;
        double ips = 1e-5;
        int gs_iter = 0.0;
        dif_lambda = 0.0;

        est = pro_WS(y_noisy, true_signal, phi, Ds, DsDs, n, lambda, b, sigma2, mse, like, s, diff, ips, gs_iter);

        if ( i % (iter / 10) == 0 || i == iter-1 || i == 0){        
            std::cout << "Iter " << gs_iter 
            << " lambda = " << lambda
            << " b = " << b 
            << " sigma2 = " << sigma2  
            << std::endl;
        }

        dif_lambda = lambda - lambda_i;
        MSE << std::fixed << std::setprecision(15) << lambda_i << "," << lambda << "," << mse << "," << like << "," << dif_lambda << "," << b << "\n";
        if(like > like_max) {
            like_max = like;
            like_max_lambda = lambda;
            like_max_mse = mse;
            like_max_est = est;
        }

        if(mse < mse_old){
            mse_old = mse;
            best_mse_lambda = lambda;
            mse_min_mse = mse;
            mse_min_est = est;
        }
        lambda_i += add;
    }
    MSE.close();

    for(int i=0; i < n; i++) true_noisy_mse += (true_signal[i] - y_noisy[i]) * (true_signal[i] - y_noisy[i]);
    true_noisy_mse /= n;

    std::cout << "--- Bayesian Optimization Finished ---" << std::endl;

    std::cout 
    << " like_max = " << like_max
    << " like_max_lambda = " << like_max_lambda 
    << " like_max_mse = " << like_max_mse 
    << " best_mse_lambda = " << best_mse_lambda 
    << " mse_min_mse = " << mse_min_mse 
    << " true_noisy_mse = " << true_noisy_mse 
    << std::endl;

    std::ofstream cnt("central.csv");
    cnt << "x,true_signal,y_noisy,estimated_signal,est_first\n";
    for (int i = 0; i < n; i++)
    {
        cnt << x[i] << "," << true_signal[i] << "," << y_noisy[i] << "," << est[i] << "," << est_first[i] << "\n";
    }
    cnt.close();

    // double super_lambda = 100000.0;
    // like_max_est = BWS2(y_noisy, true_signal, n, super_lambda, b, sigma2, mse, like);

    // 結果の出力
    std::string filename = std::to_string((int)s) + "order_result.csv";
    std::ofstream ofs(filename);
    ofs << "x,true_signal,y_noisy,estimated_signal,like_max_est,mse_min_est\n";
    for (int i = 0; i < n; i++)
    {
        ofs << x[i] << "," << true_signal[i] << "," << y_noisy[i] << "," << est[i] << "," << like_max_est[i] << "," << mse_min_est[i] << "\n";
    }   
    ofs.close();

    // std::ofstream ofs_python("python_2order_result.csv");
    // ofs_python << "x,true_signal,y_noisy,estimated_signal\n";
    // for (int i = 0; i < n; i++)
    // {
    //     ofs_python << x[i] << "," << true_signal[i] << "," << y_noisy[i] << "," << est[i] << "\n";
    // }
    // ofs_python.close();
    // std::cout << "Completed!!" << std::endl;

    // printMatrix(Ds);
    // printMatrix(DsDs);
    
    // 終了時刻を取得
    auto end = std::chrono::high_resolution_clock::now();

    // 経過時間を計算
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}