// Test doublets generator with fixed overlap

#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1
#include <iostream>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <random>
#include <armadillo>

using namespace std;


// Parameters: N = #dimensions, P = #couples, rho = overlap, seed = random seed
vector<arma::vec> randCouples_arma_rng(int N, int P, double rho, int seed) {

    // Set RNG
    arma::arma_rng::set_seed(seed);

    vector<arma::vec> xibar;
    arma::vec xi(N, arma::fill::zeros);
    xi(N-1) = 1.;


    for (int mu = 0; mu < P; mu++) {
        arma::vec bar(N-1, arma::fill::randn);

        bar = (bar/norm(bar, 2))*sqrt(1. - rho*rho);
        bar.resize(N);
        bar(N-1) = rho;


        arma::mat randmat(N, N, arma::fill::randn);
        arma::mat Q, R;

        bool is_completed = arma::qr(Q, R, randmat);

        if (!is_completed) {
            cerr << "ERROR: QR decomposition failed on step " << mu + 1 << endl;
            exit(1);
        }

        int sign = (arma::randu() < 0.5) ? -1 : 1;
        Q *= sign;

        xibar.push_back(Q * xi);
        xibar.push_back(Q * bar);
    }

    return xibar;
}


// Parameters: N = #dimensions, P = #couples, rho = overlap, seed = random seed
vector<arma::vec> randCouples(int N, int P, double rho, int seed) {

    // Set RNG
    mt19937 generator(seed);
    normal_distribution<double> rand_normal(0., 1.);
    uniform_real_distribution<double> rand_uniform(0., 1.);

    vector<arma::vec> xibar;
    arma::vec xi(N, arma::fill::zeros);
    xi(N-1) = 1.;

    // Prepare couples
    for (int mu = 0; mu < P; mu++) {
        arma::vec bar(N-1);

        for (int i = 0; i < N-1; i++)
            bar(i) = rand_normal(generator);

        bar = (bar/norm(bar, 2)) * sqrt(1. - rho*rho);
        bar.resize(N);
        bar(N-1) = rho;

        arma::mat randmat(N, N);
        arma::mat Q, R;

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                randmat(i,j) = rand_normal(generator);
        }

        bool is_completed = arma::qr(Q, R, randmat);

        if (!is_completed) {
            cerr << "QR decomposition failed on step " << mu + 1 << endl;
            exit(1);
        }

        int sign = (rand_uniform(generator) < 0.5) ? -1 : 1;
        Q *= sign;

        xibar.push_back(Q*xi);
        xibar.push_back(Q*bar);
    }

    return xibar;
}


// Parameters: N = #dimensions, R = radius of the hypersphere
double distribution_mean(int N, double R) {

    return pow(2., N-1) * R * beta(N/2., N/2.) / beta(N - 0.5, 0.5);
}


// Parameters: N = #dimensions, R = radius of the hypersphere
double distribution_variance(int N, double R) {

    double num = pow(2., 2*N - 2) * pow(tgamma(N/2.), 4);
    double den = M_PI * pow(tgamma(N - 0.5), 2);

    return (2. - (num/den))*R*R;
}


/*********************************   MAIN   **********************************/
int main(int argc, char const *argv[]) {

    if (argc < 5 || argc > 6) {
        cerr << endl << "Wrong  call " << argv[0] << endl;
        cerr << "Required arguments for execution are: <N_min> <N_max> <P> <rho> (<rng seed> optional)" << endl << endl;
        return 1;
    }

    const int N_min = atoi(argv[1]);
    const int N_max = atoi(argv[2]);
    const int P = atoi(argv[3]);
    const float rho = atof(argv[4]);
    int seed = time(nullptr);
    
    if (argc == 6)
    	seed = atoi(argv[5]);

    int n_couples = P*(2*P - 1);


    // Run test
    cout << "Dimension" << setw(7) << "Mean" << setw(18) << "Calc_mean";
    cout << setw(12) << "Variance" << setw(18) << "Calc_variance" << endl;

    for (int i = N_min; i < N_max; i++) {
        auto v = randCouples(i, P, rho, seed);
        arma::vec dist_vec(n_couples, arma::fill::zeros);
        double cumulative_sum = 0., var_calculation = 0.;
        int index = 0;
        int c_param = (i >= 10) ? 1 : 0;

        for (int j = 0; j < 2*P; j++) {
            for (int k = j + 1; k < 2*P; k++) {
                double appo = sqrt(arma::as_scalar((v[j] - v[k]).t() * (v[j] - v[k])));
                dist_vec(index) = appo;

                cumulative_sum += dist_vec(index);
                index++;
            }
        }

        double ave_distance = cumulative_sum / n_couples;

        for (auto dist : dist_vec)
            var_calculation += pow(dist - ave_distance, 2);

        var_calculation /= n_couples - 1.;

	    cout.precision(6);
        cout << i << setw(19 - c_param) << fixed << distribution_mean(i,1.) << setw(13) << ave_distance;
        cout << setw(13) << distribution_variance(i,1.) << setw(13) << var_calculation << endl;
    }
    

    return 0;
}
