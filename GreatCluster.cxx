vector<float> GreatCluster(const vector<vector<float>>& Shapes)
{
    //cout<<"NEW GREAT CLUSTER"<<endl;
    const int IN_PARAMS = 11;
    const int OUT_PARAMS = 7;

    const int ENERGY = 7;
    const int POS_X  = 8;
    const int POS_Y  = 9;
    const int POS_Z  = 10;

    vector<float> out(OUT_PARAMS, 0.0f);
    if (Shapes.empty()) return out;
    //cout<<"NOT EMPTY"<<endl;

    for (auto& s : Shapes)
        if ((int)s.size() < IN_PARAMS) return out;
    //cout<<"GOOD NUMBER"<<endl;

    float w_sum = 0.0f;
    for (auto& s : Shapes) w_sum += s[ENERGY];
    if (w_sum == 0) return out;
    //cout<<"GOOD ENERGY"<<endl;


    int n = Shapes.size();
    //cout<<"Number of Shapes: "<<n<<endl;
    // ============================================================
    // SPECIAL CASE: only one cluster → return its local parameters
    // ============================================================
    if (n == 1) {

        const auto& s = Shapes[0];

        out[0] = s[0];   // radius
        out[1] = s[1];   // dispersion

        // 2D eigenvalues — SORT
        {
            std::vector<float> e2 = { s[2], s[3] };
            std::sort(e2.begin(), e2.end());
            out[2] = e2[0];
            out[3] = e2[1];
        }

        // 3D eigenvalues — SORT
        {
            std::vector<float> e3 = { s[4], s[5], s[6] };
            std::sort(e3.begin(), e3.end());
            out[4] = e3[0];
            out[5] = e3[1];
            out[6] = e3[2];
        }

       // cout<<"0: "<<out[0]<<" i 1: "<<out[1]<<endl;
        //cout<<"2: "<<out[2]<<" i 3: "<<out[3]<<endl;
        //cout<<"4: "<<out[4]<<" i 5: "<<out[5]<<" i 6: "<<out[6]<<endl;
        //cout<<"============================"<<endl;


        return out;
    }

    // ============================================================
    // 1) Center of Mass
    // ============================================================
    float cx=0, cy=0, cz=0;
    for (auto& s : Shapes) {
        float w = s[ENERGY];
        cx += w * s[POS_X];
        cy += w * s[POS_Y];
        cz += w * s[POS_Z];
    }
    cx /= w_sum;
    cy /= w_sum;
    cz /= w_sum;

    // ============================================================
    // 2) Radius and Dispersion
    // ============================================================
    float sum_r2_unw = 0.0f;
    float sum_r2_w   = 0.0f;

    for (auto& s : Shapes) {
        float w = s[ENERGY];

        float dx = s[POS_X] - cx;
        float dy = s[POS_Y] - cy;
        float dz = s[POS_Z] - cz;
        float r2 = dx*dx + dy*dy + dz*dz;

        sum_r2_unw += r2;
        sum_r2_w   += r2 * w;
    }

    out[0] = sqrt(sum_r2_unw / max(1, n - 1)); // radius
    out[1] = sqrt(sum_r2_w / w_sum);           // dispersion

    // ============================================================
    // 3) Macierz kowariancji 3D
    // ============================================================
    double Cxx3=0, Cyy3=0, Czz3=0, Cxy3=0, Cxz3=0, Cyz3=0;

    for (auto& s : Shapes) {
        float w = s[ENERGY];
        float dx = s[POS_X] - cx;
        float dy = s[POS_Y] - cy;
        float dz = s[POS_Z] - cz;

        Cxx3 += w * dx * dx;
        Cyy3 += w * dy * dy;
        Czz3 += w * dz * dz;
        Cxy3 += w * dx * dy;
        Cxz3 += w * dx * dz;
        Cyz3 += w * dy * dz;
    }

    Cxx3 /= w_sum; Cyy3 /= w_sum; Czz3 /= w_sum;
    Cxy3 /= w_sum; Cxz3 /= w_sum; Cyz3 /= w_sum;

    // ============================================================
    // 4) Eigenvalues 3D
    // ============================================================
    TMatrixDSym cov3sym(3);
    cov3sym(0,0) = Cxx3;  cov3sym(0,1) = Cxy3;  cov3sym(0,2) = Cxz3;
    cov3sym(1,0) = Cxy3;  cov3sym(1,1) = Cyy3;  cov3sym(1,2) = Cyz3;
    cov3sym(2,0) = Cxz3;  cov3sym(2,1) = Cyz3;  cov3sym(2,2) = Czz3;

    TMatrixDSymEigen eig3(cov3sym);
    TVectorD evals3 = eig3.GetEigenValues();

    std::vector<double> e3_sorted = { evals3[0], evals3[1], evals3[2] };
    std::sort(e3_sorted.begin(), e3_sorted.end());

    out[4] = e3_sorted[0];
    out[5] = e3_sorted[1];
    out[6] = e3_sorted[2];

    // ============================================================
    // 5) Eigenvalues 2D (theta-phi)
    // ============================================================
    double Sx=0, Sy=0, Sxx=0, Sxy=0, Syy=0;
    double wsum2D = 0;

    for (auto& s : Shapes) {
        double w = s[ENERGY];
        TVector3 pos(s[POS_X], s[POS_Y], s[POS_Z]);

        double theta = pos.Theta();
        double phi   = pos.Phi();

        Sx  += w * theta;
        Sy  += w * phi;
        Sxx += w * theta * theta;
        Sxy += w * theta * phi;
        Syy += w * phi   * phi;

        wsum2D += w;
    }

    Sx  /= wsum2D;
    Sy  /= wsum2D;
    Sxx /= wsum2D;
    Sxy /= wsum2D;
    Syy /= wsum2D;

    double Cxx2 = Sxx - Sx * Sx;
    double Cxy2 = Sxy - Sx * Sy;
    double Cyy2 = Syy - Sy * Sy;

    TMatrixDSym cov2(2);
    cov2(0,0) = Cxx2;
    cov2(0,1) = Cxy2;
    cov2(1,0) = Cxy2;
    cov2(1,1) = Cyy2;

    TMatrixDSymEigen eig2(cov2);
    TVectorD evals2 = eig2.GetEigenValues();

    std::vector<double> e2_sorted = { evals2[0], evals2[1] };
    std::sort(e2_sorted.begin(), e2_sorted.end());

    out[2] = e2_sorted[0];
    out[3] = e2_sorted[1];

    //cout<<"0: "<<out[0]<<" i 1: "<<out[1]<<endl;
    //cout<<"2: "<<out[2]<<" i 3: "<<out[3]<<endl;
    //cout<<"4: "<<out[4]<<" i 5: "<<out[5]<<" i 6: "<<out[6]<<endl;
    //cout<<"============================"<<endl;

    return out;
}
