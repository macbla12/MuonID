vector<float> GreatCluster(const vector<vector<float>>& Shapes)
{
    const int IN_PARAMS = 11;
    const int OUT_PARAMS = 7;

    const int ENERGY = 7;
    const int POS_X  = 8;
    const int POS_Y  = 9;
    const int POS_Z  = 10;

    vector<float> out(OUT_PARAMS, 0.0f);
    if (Shapes.empty()) return out;

    for (auto& s : Shapes)
        if ((int)s.size() < IN_PARAMS) return out;

    float w_sum = 0.0f;
    for (auto& s : Shapes) w_sum += s[ENERGY];
    if (w_sum == 0) return out;

    int n = Shapes.size();
    //cout<<"Number of Shapes: "<<n<<endl;

    // ============================================================
    // SPECIAL CASE: only one cluster → return its local parameters
    // ============================================================
    if (n == 1) {

        const auto& s = Shapes[0];

        // out[0] radius (local)
        out[0] = s[0];

        // out[1] dispersion (local)
        out[1] = s[1];

        // out[2], out[3] = 2D eigenvalues (local)
        out[2] = s[2];
        out[3] = s[3];

        // out[4], out[5], out[6] = 3D eigenvalues (local)
        out[4] = s[4];
        out[5] = s[5];
        out[6] = s[6];
        //cout<<"0: "<<out[0]<<" i 1: "<<out[1]<<endl;
        //cout<<"2: "<<out[2]<<" i 3: "<<out[3]<<endl;
        //cout<<"4: "<<out[4]<<" i 5: "<<out[5]<<" i 6: "<<out[6]<<endl;

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
    float max_dist   = 0.0f;

    
    float sum_local_disp_w = 0.0f;
    float max_local_eigen3D = 0.0f;

    for (auto& s : Shapes) {
        float w = s[ENERGY];

        float dx = s[POS_X] - cx;
        float dy = s[POS_Y] - cy;
        float dz = s[POS_Z] - cz;
        float r2 = dx*dx + dy*dy + dz*dz;

        sum_r2_unw += r2;
        sum_r2_w   += r2 * w;

        float dist = sqrt(r2);
        if (dist > max_dist) max_dist = dist;

        // lokalne shape’y
        float local_disp = s[1];
        sum_local_disp_w += w * local_disp;

        float local_e3 = s[6]; // największa 3D eigenvalue
        if (local_e3 > max_local_eigen3D) max_local_eigen3D = local_e3;
    }
    

    out[0] = sqrt(sum_r2_unw / max(1, n - 1));
    out[1] = sqrt(sum_r2_w / w_sum);
    // ============================================================
    // 3) Macierz kowariancji 3D
    // ============================================================
    float Cxx=0, Cyy=0, Czz=0, Cxy=0, Cxz=0, Cyz=0;

    for (auto& s : Shapes) {
        float w = s[ENERGY];
        float dx = s[POS_X] - cx;
        float dy = s[POS_Y] - cy;
        float dz = s[POS_Z] - cz;

        Cxx += w * dx * dx;
        Cyy += w * dy * dy;
        Czz += w * dz * dz;
        Cxy += w * dx * dy;
        Cxz += w * dx * dz;
        Cyz += w * dy * dz;
    }

    Cxx /= w_sum; Cyy /= w_sum; Czz /= w_sum;
    Cxy /= w_sum; Cxz /= w_sum; Cyz /= w_sum;

    // ============================================================
    // 4) Eigenvalues 3D (tu wstawiasz swoją diagonalizację)
    // ============================================================
    // zbuduj macierz symetryczną 3×3
    TMatrixDSym cov3sym(3);
    cov3sym(0,0) = Cxx;  cov3sym(0,1) = Cxy;  cov3sym(0,2) = Cxz;
    cov3sym(1,0) = Cxy;  cov3sym(1,1) = Cyy;  cov3sym(1,2) = Cyz;
    cov3sym(2,0) = Cxz;  cov3sym(2,1) = Cyz;  cov3sym(2,2) = Czz;

    // diagonalizacja
    TMatrixDSymEigen eig3(cov3sym);

    // wartości własne (REAL)
    TVectorD evals3 = eig3.GetEigenValues();

    // posortuj rosnąco
    std::vector<double> e3_sorted = { evals3[0], evals3[1], evals3[2] };
    std::sort(e3_sorted.begin(), e3_sorted.end());

    // zapisz
    out[4] = e3_sorted[0];
    out[5] = e3_sorted[1];
    out[6] = e3_sorted[2];

   // ============================================================
    // 5) Eigenvalues 2D (theta-phi) — EXACT EIC LOGIC
    // ============================================================

    // Suma wag
    double wsum2D = 0;

    // sumy pierwsze i drugie
    double Sxx2 = 0, Sxy2 = 0, Syy2 = 0;
    double Sx2 = 0, Sy2 = 0;

    for (auto& s : Shapes) {

        double w = s[ENERGY];

        // konwersja pozycji 3D → theta, phi
        TVector3 pos(s[POS_X], s[POS_Y], s[POS_Z]);

        double theta = pos.Theta();   // = anglePolar
        double phi   = pos.Phi();     // = angleAzimuthal

        // sumy pierwsze
        Sx2 += w * theta;
        Sy2 += w * phi;

        // sumy drugie
        Sxx2 += w * theta * theta;
        Sxy2 += w * theta * phi;
        Syy2 += w * phi * phi;

        wsum2D += w;
    }

    // normalizacja
    Sx2  /= wsum2D;
    Sy2  /= wsum2D;
    Sxx2 /= wsum2D;
    Sxy2 /= wsum2D;
    Syy2 /= wsum2D;

    // macierz kowariancji
    double Cxx2 = Sxx2 - Sx2 * Sx2;
    double Cxy2 = Sxy2 - Sx2 * Sy2;
    double Cyy2 = Syy2 - Sy2 * Sy2;

    // policz wartości własne 2×2
    double T = Cxx2 + Cyy2;
    double D = Cxx2 * Cyy2 - Cxy2 * Cxy2;
    double disc = std::sqrt(std::max(0.0, T*T/4 - D));

    double lambda1 = T/2 + disc;
    double lambda2 = T/2 - disc;

    // zapis
    out[2] = lambda1;
    out[3] = lambda2;   
    //cout<<"0: "<<out[0]<<" i 1: "<<out[1]<<endl;
    //cout<<"2: "<<out[2]<<" i 3: "<<out[3]<<endl;
    //cout<<"4: "<<out[4]<<" i 5: "<<out[5]<<" i 6: "<<out[6]<<endl;
    


    return out;
}
