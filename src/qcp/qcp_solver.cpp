// Run QCP using Gurobi optimizer

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <string>
#include <armadillo>
#include <gurobi_c++.h>

using namespace std;


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


/*********************************   MAIN   **********************************/
int main(int argc, char const *argv[]) {

    if (argc != 8) {
        cerr << endl << "Wrong  call " << argv[0] << endl;
        cerr << "Required arguments for execution are: <N> <P> <rho> <seed> <gurobi_seed> <constraint_name> <epsilon>" << endl << endl;
        return 1;
    }

    const int N = atoi(argv[1]);
    const int P = atoi(argv[2]);
    const float rho = atof(argv[3]);
    const int seed = atoi(argv[4]);
    const int gseed = atoi(argv[5]);
    string constraint = argv[6];
    const float eps = atof(argv[7]);

    if (constraint != "weps" && constraint != "w2eps" && constraint != "wnorm") {
        cerr << "ERROR: <constraint_name> must be one of weps/w2eps/wnorm" << endl;
        exit(1);
    }

    // Generate doublets
    auto doublets = randCouples(N,P,rho,seed);

    try {
        GRBEnv env(true);
        env.start();

        GRBModel model(env);

        // Create optimization variables
        GRBVar* w = model.addVars(N, GRB_CONTINUOUS);
        for (int i = 0; i < N; i++)
                w[i].set(GRB_DoubleAttr_LB, -GRB_INFINITY);

        // Add constraint
        if (constraint == "weps") {
            model.addConstr(w[0] >= eps);
        }
        else if (constraint == "w2eps") {
            model.addQConstr(w[0]*w[0] >= eps);
        }
        else {
            GRBQuadExpr w_norm = 0.;
            vector<double> v_one(N);
            fill(v_one.begin(), v_one.end(), 1.);

            w_norm.addTerms(v_one.data(), w, w, N);
            model.addQConstr(w_norm >= eps);
        }

        // Constraints (on doublets)
        for (int i = 0; i < 2*P; i += 2) {
            GRBLinExpr appo_xi = 0., appo_xibar = 0.;

            appo_xi.addTerms(doublets[i].memptr(), w, N);
            appo_xibar.addTerms(doublets[i+1].memptr(), w, N);

            model.addQConstr(appo_xi*appo_xibar >= 0);
        }

        // Set objective function (constant)
        model.setObjective(1 + 0*w[0], GRB_MINIMIZE);

        // Set parameters for optimization
        model.set(GRB_IntParam_NonConvex, 2);              // Compute non-convex QCP
	    model.set(GRB_IntParam_MIPFocus, 1);               // Focus on finding a solution (even thoungh non optimal)
        model.set(GRB_IntParam_Threads, 4);	               // Set max number of threads during parallel algorithms
        model.set(GRB_IntParam_Seed, gseed);               // Set seed for internal RNG (default 0)
        model.set(GRB_IntParam_DisplayInterval, 30);       // Frequency of log lines
	    model.set(GRB_DoubleParam_TimeLimit, 14400);       // Max time for optimization
        model.set(GRB_DoubleParam_Heuristics, 0.15);       // Fraction of total time spent in feasibility heuristics
        model.set(GRB_DoubleParam_FeasibilityTol, 1e-8);   // Primal feasibility tolerance (default 10^-6) (in [10^-9,10^-2])

        // Print list of parameters
        cout << endl << "List of input parameters:" << endl;
        cout << "# N" << setw(16) << "|" << N << endl;
        cout << "# P" << setw(16) << "|" << P << endl;
        cout << "# rho" << setw(14) << "|" << rho << endl;
        cout << "# seed" << setw(13) << "|" << seed << endl;
        cout << "# epsilon" << setw(10) << "|" << eps << endl;
        cout << "# constraint" << setw(7) << "|" << constraint << endl;
        cout << "# gurobi_seed" << setw(6) << "|" << model.get(GRB_IntParam_Seed) << endl;
        cout << "# feasibilityTol" << setw(3) << "|" << model.get(GRB_DoubleParam_FeasibilityTol) << endl;
        cout << endl;

        // Run optimization
        model.optimize();

        // Check results
        if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
            cout << endl << "Objective function: " << model.get(GRB_DoubleAttr_ObjVal) << endl;
            cout << "W components: ";
            for (int i = 0; i < N; i++)
                cout << w[i].get(GRB_DoubleAttr_X) << " ";
            cout << endl << endl;

            // Check constraint satisfaction
            arma::rowvec w_tmp(N);
            double sp1 = 0., sp2 = 0.;
            bool is_positive_strict = true, is_positive_loose = true;

            for (int i = 0; i < N; i++)
                w_tmp(i) = w[i].get(GRB_DoubleAttr_X);

            cout << "Csi*W and Csibar*W:" << endl;

            for (int i = 0; i < 2*P; i += 2) {
                sp1 = arma::as_scalar(w_tmp*doublets[i]);
                sp2 = arma::as_scalar(w_tmp*doublets[i+1]);

		        cout << sp1 << " " << sp2 << endl;

                if (sp1*sp2 <= 0)
                    is_positive_strict = false;

                if (sp1*sp2 < 0)
                    is_positive_loose = false;
            }

            cout << endl << "Constraint satisfaction check:" << endl;
            cout << "Strict (>): " << (is_positive_strict ? "true" : "false") << endl;
            cout << "Loose (>=): " << (is_positive_loose  ? "true" : "false") << endl;
        }
        else {
            cout << "Non Optimal result. Status Code: " << model.get(GRB_IntAttr_Status) << endl;

            if (model.get(GRB_IntAttr_SolCount) == 0) {
                cout << "No solution found during optimization." << endl;
            }
            else {
                cout << "Number of solutions: " << model.get(GRB_IntAttr_SolCount) << endl;
            }
        }

        delete [] w;

    }
    catch (GRBException e) {
        cerr << "Error code = " << e.getErrorCode() << endl;
        cerr << e.getMessage() << endl;
    }
    catch (...) {
       cerr << "Unknown exception caught" << endl;
    }


    return 0;
}
