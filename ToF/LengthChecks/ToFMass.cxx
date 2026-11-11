// ToFMass.cxx  -  tylko ToF: rozklad masy + diagnostyka (Grape vs single)
// Uruchom: root -l -q ToFMass.cxx++
//
// Dlugosc toru: CentralTrackSegments (pt.pathlength), najblizszy punkt segmentu do hitu ToF.
// Beta: fit t = L/(beta c) przez zero (jak w oryginale).

#include <TH1.h>
#include <TH2.h>
#include <TProfile.h>
#include <TF1.h>
#include <TFile.h>
#include <TROOT.h>
#include <TVector3.h>
#include <TVector2.h>
#include <TLorentzVector.h>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <glob.h>

#include <podio/Frame.h>
#include <podio/ROOTReader.h>
#include <edm4eic/ReconstructedParticleCollection.h>
#include <edm4eic/TrackerHitCollection.h>
#include <edm4eic/TrackSegmentCollection.h>

using namespace std;

static constexpr double c_light = 299.792458; // mm/ns

// okna geometryczne ToF [mm]
// BTOF: 63<R<66 cm, |z|<120 cm;  FTOF: 8<R<67 cm, 180<|z|<190 cm (+ zapas na tilt/gruboszc)
const double BTOF_RMIN = 610,  BTOF_RMAX = 680,  BTOF_ZMAX = 1250;
const double FTOF_ZMIN = 1750, FTOF_ZMAX = 1950, FTOF_RMIN = 70, FTOF_RMAX = 680;

static bool InBarrel(double x, double y, double z) {
    double R = hypot(x, y), az = fabs(z);
    return R >= BTOF_RMIN && R <= BTOF_RMAX && az <= BTOF_ZMAX;
}
static bool InEndcap(double x, double y, double z) {
    double R = hypot(x, y), az = fabs(z);
    return az >= FTOF_ZMIN && az <= FTOF_ZMAX && R >= FTOF_RMIN && R <= FTOF_RMAX;
}

// Korekta t0: z wykresu dT vs vz wynika t_hit - L/c ~ -vz/c  =>  t0 = -vz/c  (sprawdz nachylenie w wydruku!)
const bool   CORRECT_T0 = true;  // czy korekta t0 = -vz/c
const double T0_PER_MM  = -1.0 / c_light;   // t0 = T0_PER_MM * vz   [ns/mm]
// Odrzucamy punkty segmentu, dla ktorych pathlength < odleglosc w linii prostej od wierzcholka (fizycznie niemozliwe)
const double L_TOL = 10.0;                  // [mm]

struct Pt { double x, y, z, path; };
using TL = vector<pair<double,double>>; // {time, length}

// beta z fitu t = L/(c*beta) przez zero; zwraca <=0 gdy brak
double CombineBeta(const TL &h) {
    double sxx = 0, sxt = 0;
    for (auto &p : h) { double x = p.second / c_light; sxx += x*x; sxt += x*p.first; }
    if (sxx <= 0 || sxt <= 0) return -1;
    return sxx / sxt;
}

vector<string> Glob(const string &pat) {
    vector<string> r; glob_t g;
    glob(pat.c_str(), GLOB_TILDE, nullptr, &g);
    for (size_t i = 0; i < g.gl_pathc; ++i) r.push_back(g.gl_pathv[i]);
    globfree(&g);
    return r;
}

const Pt *Nearest(const vector<Pt> &pts, const TVector3 &h, double &d) {
    const Pt *b = nullptr; d = 1e18;
    for (auto &p : pts) {
        double x = sqrt(pow(p.x-h.X(),2) + pow(p.y-h.Y(),2) + pow(p.z-h.Z(),2));
        if (x < d) { d = x; b = &p; }
    }
    return b;
}

// Dopasowanie hitow jednego detektora (barrel albo endcap) do tracka
void Match(const edm4eic::TrackerHitCollection &hits, const vector<Pt> &pts,
           const TLorentzVector &P, double dRcut, double distCut, TL &out,
           TH1D *hDR, TH1D *hDist, TH1D *hDt, TH2D *hTvsL, TH1D *hHitBeta,
           TH2D *hDtVsVz, double vz, const TVector3 &ref, TH1D *hLdiff, TH2D *hLvsEu, TH1D *hDtCorr)
{
    for (const auto &hit : hits) {
        auto p = hit.getPosition();
        TVector3 pos(p.x, p.y, p.z);
        double dEta = P.Eta() - pos.Eta();
        double dPhi = TVector2::Phi_mpi_pi(P.Phi() - pos.Phi());
        double dR = sqrt(dEta*dEta + dPhi*dPhi);
        hDR->Fill(dR);
        if (dR >= dRcut) continue;

        double d; const Pt *np = Nearest(pts, pos, d);
        if (!np) continue;
        hDist->Fill(d);
        if (d >= distCut) continue;

        double tRaw = hit.getTime(), L = np->path;
        double t = tRaw - (CORRECT_T0 ? T0_PER_MM * vz : 0.0);   // czas po korekcie t0
        out.push_back({t, L});
        // kontrola: pathlength (po torze) vs linia prosta od punktu referencyjnego do punktu segmentu
        double Leu = (TVector3(np->x, np->y, np->z) - ref).Mag();
        hLdiff->Fill(L - Leu);   // powinno byc >= 0 (luk >= cieciwa)
        hLvsEu->Fill(Leu, L);    // powinno lezec nieco powyzej przekatnej
        hDt->Fill(tRaw - L / c_light);           // SUROWY czas; dla beta~1 oczekiwane ~0
        hDtCorr->Fill(t - L / c_light);          // po korekcie t0
        hTvsL->Fill(L, t);                       // linia t = L/c
        if (t > 0) hHitBeta->Fill(L / c_light / t);
        hDtVsVz->Fill(vz, tRaw - L / c_light);   // surowy: nachylenie = -1/c => t0 = -vz/c
    }
}

struct Set { string name, pattern; };

void ToFMass()
{
    gROOT->SetBatch(kTRUE);

    vector<Set> sets = {
        {"SingleMuons", "/run/media/epic/Data/Background/Muon/Podio-1.7.99/reco_*LowP*.root"},
        {"SinglePions", "/run/media/epic/Data/Background/Pion/Podio-1.7.99/reco_*LowP*.root"},

    };

    const double dRcut = 0.8, distCut = 6.0; // dR i odleglosc hit<->segment [mm]
    const double pMax = 2.0;                 // jak w starym IDAnalysis (p<1 GeV); sigma(m^2) rosnie jak p^2! ustaw 100 zeby zobaczyc wszystko
    const long   maxEv = -1;              // -1 = wszystkie

    for (auto &s : sets) {
        auto files = Glob(s.pattern);
        if (files.empty()) { cerr << "[ERR] brak plikow: " << s.pattern << endl; continue; }

        podio::ROOTReader reader;
        reader.openFiles(files);
        long nEnt = reader.getEntries("events");
        long nProc = (maxEv > 0) ? min(maxEv, nEnt) : nEnt;

        TFile out(Form("Plots/ToFMass_%s.root", s.name.c_str()), "RECREATE");
        auto N = [&](const char *n) { return Form("%s_%s", n, s.name.c_str()); };

        // --- masa (ze znakiem: m = sign(m^2)*sqrt(|m^2|), zeby nie gubic m^2<0) ---
        auto *hM     = new TH1D(N("Mass"),        "mass (all ToF);m [GeV];Counts", 200, 0, 0.3);
        auto *hMB    = new TH1D(N("Mass_Barrel"), "mass barrel only;m [GeV];Counts", 200, 0, 0.3);
        auto *hME    = new TH1D(N("Mass_Endcap"), "mass endcap only;m [GeV];Counts", 200, 0, 0.3);
        auto *hMPos  = new TH1D(N("Mass_Pos"),    "mass charge +;m [GeV];Counts", 200, 0, 0.3);
        auto *hMNeg  = new TH1D(N("Mass_Neg"),    "mass charge -;m [GeV];Counts", 200, 0, 0.3);
        // rozdzielczosc masy silnie zalezy od pedu (sigma(m^2) ~ p^2) -> osobne przedzialy p
        TH1D *hMp[6] = {
            new TH1D(N("Mass_p0"),  "mass, p<0.3 GeV;m [GeV];Counts",   200, 0, 0.3),
            new TH1D(N("Mass_p1"),  "mass, 0.3<p<0.5 GeV;m [GeV];Counts",   200, 0, 0.3),
            new TH1D(N("Mass_p2"),  "mass, 0.5<p<0.7 GeV;m [GeV];Counts", 200, 0, 0.3),
            new TH1D(N("Mass_p3"),  "mass, 0.7<p<0.9 GeV;m [GeV];Counts",   200, 0, 0.3),
            new TH1D(N("Mass_p4"),  "mass, 0.9<p<1.1 GeV;m [GeV];Counts",   200, 0, 0.3),
            new TH1D(N("Mass_p5"),  "mass, p>1.1 GeV;m [GeV];Counts", 200, 0, 0.3)};
        auto *hMvsP  = new TH2D(N("MassVsP"),     "mass vs p;p [GeV];m [GeV]", 110, 0, 1.1, 200, 0, 0.3);
        auto *hBetaP = new TH2D(N("BetaVsP"),     "#beta vs p;p [GeV];#beta", 100, 0, 2, 100, 0.8, 1.1);

        auto *hLdiff = new TH1D(N("Lpath_minus_Leucl"), "L_{path} - L_{eucl};mm;Counts", 200, -50, 150);
        auto *hLvsEu = new TH2D(N("Lpath_vs_Leucl"), "L_{path} vs L_{eucl};L_{eucl} [mm];L_{path} [mm]", 100, 0, 4000, 100, 0, 4000);

        // --- diagnostyka ---
        auto *hDRB   = new TH1D(N("dR_Barrel"),  "dR barrel;#DeltaR;Counts", 200, 0, 2);
        auto *hDRE   = new TH1D(N("dR_Endcap"),  "dR endcap;#DeltaR;Counts", 200, 0, 2);
        auto *hDistB = new TH1D(N("Dist_Barrel"), "dist hit-segment barrel;mm;Counts", 200, 0, 100);
        auto *hDistE = new TH1D(N("Dist_Endcap"), "dist hit-segment endcap;mm;Counts", 200, 0, 100);
        auto *hDt    = new TH1D(N("dT_hit_minus_Lc"), "t_{hit} - L/c;ns;Counts", 300, -1, 1);
        auto *hTvsL  = new TH2D(N("TimeVsLength"), "t_{hit} vs L;L [mm];t [ns]", 100, 0, 4000, 100, 0, 10);
        auto *hHitB  = new TH1D(N("HitBeta"), "per-hit #beta = L/(c t);#beta;Counts", 200, 0.8, 1.1);
        auto *hDtCorr = new TH1D(N("dT_corrected"), "t_{hit}-t_{0} - L/c (po korekcie t0);ns;Counts", 300, -1, 1);
        auto *hDtVz  = new TH2D(N("dTVsVz"), "t-L/c vs track vz;vz [mm];t-L/c [ns]", 100, -300, 300, 100, -1, 1);
        auto *hNRcp  = new TH1D(N("NRecoPerEvent"), "N ReconstructedParticles / event;N;Events", 20, -0.5, 19.5);
        auto *hNHits = new TH1D(N("NHitsMatched"), "N matched ToF hits / track;N;Tracks", 20, -0.5, 19.5);
        auto *hP     = new TH1D(N("TrackP"), "track p;p [GeV];Tracks", 100, 0, 2);

        // --- kontrola polozenia: gdzie naprawde sa powierzchnie segmentu i hity ToF ---
        auto *hSegR   = new TH1D(N("SegPoint_R_all"), "R of ALL CentralTrackSegments points;R [mm];Points", 500, 0, 1000);
        auto *hSegRZ  = new TH2D(N("SegPoint_RvsZ_all"), "segment points: R vs z;z [mm];R [mm]", 300, -2500, 2500, 200, 0, 1000);
        auto *hBHitR  = new TH1D(N("BTOFHit_R"), "TOFBarrelRecHits R;R [mm];Hits", 200, 550, 750);
        auto *hBHitZ  = new TH1D(N("BTOFHit_Z"), "TOFBarrelRecHits z;z [mm];Hits", 300, -1500, 1500);
        auto *hEHitZ  = new TH1D(N("FTOFHit_Z"), "TOFEndcapRecHits |z|;|z| [mm];Hits", 300, 1500, 2200);
        auto *hEHitR  = new TH1D(N("FTOFHit_R"), "TOFEndcapRecHits R;R [mm];Hits", 200, 0, 800);

        long nTrk = 0, nNoTrack = 0, nSegPts = 0, nHit = 0, nBeta = 0, nBadPath = 0;

        vector<std::string> collections = {
            "ReconstructedParticles", 
            "TOFBarrelSharedRecHits", 
            "TOFEndcapSharedRecHits", 
            "CentralTrackSegments",
            "LFHCALClusters"          
        };

        for (long e = 0; e < nProc; ++e) {
            podio::Frame frame(reader.readEntry("events", e, collections));
            auto names = frame.getAvailableCollections();
            auto has = [&](const string &n) { return find(names.begin(), names.end(), n) != names.end(); };
            if (!has("ReconstructedParticles") || !has("CentralTrackSegments") ||
                !has("TOFBarrelSharedRecHits") || !has("TOFEndcapSharedRecHits")) {
                if (e == 0) cerr << "[WARN] brak wymaganej kolekcji w " << s.name << endl;
                continue;
            }

            const auto &rcps = frame.get<edm4eic::ReconstructedParticleCollection>("ReconstructedParticles");
            const auto &segs = frame.get<edm4eic::TrackSegmentCollection>("CentralTrackSegments");
            const auto &bH   = frame.get<edm4eic::TrackerHitCollection>("TOFBarrelSharedRecHits");
            const auto &eH   = frame.get<edm4eic::TrackerHitCollection>("TOFEndcapSharedRecHits");
            hNRcp->Fill(rcps.size());
            for (const auto &h : bH) { auto q = h.getPosition(); hBHitR->Fill(hypot(q.x, q.y)); hBHitZ->Fill(q.z); }
            for (const auto &h : eH) { auto q = h.getPosition(); hEHitR->Fill(hypot(q.x, q.y)); hEHitZ->Fill(fabs(q.z)); }

            for (const auto rcp : rcps) {
                auto m = rcp.getMomentum();
                TLorentzVector P; P.SetPxPyPzE(m.x, m.y, m.z, rcp.getEnergy());
                if (P.Eta() < -3) continue;
                double p = P.P();
                if (p <= 0 || p > pMax) continue;
                ++nTrk; hP->Fill(p);
                if (rcp.getTracks().empty()) { ++nNoTrack; continue; }
                const auto trk = rcp.getTracks().at(0);

                auto rp = rcp.getReferencePoint();
                TVector3 ref(rp.x, rp.y, rp.z);
                double vz = rp.z;

                vector<Pt> bPts, ePts;
                for (const auto seg : segs) {
                    if (seg.getTrack() != trk) continue;
                    for (const auto pt : seg.getPoints()) {
                        double x = pt.position.x, y = pt.position.y, z = pt.position.z;
                        hSegR->Fill(hypot(x, y)); hSegRZ->Fill(z, hypot(x, y));
                        if (pt.pathlength < (TVector3(x, y, z) - ref).Mag() - L_TOL) { ++nBadPath; continue; }
                        if (InBarrel(x, y, z))      bPts.push_back({x, y, z, (double)pt.pathlength});
                        else if (InEndcap(x, y, z)) ePts.push_back({x, y, z, (double)pt.pathlength});
                    }
                }
                if (!bPts.empty() || !ePts.empty()) ++nSegPts;

                TL mb, me;
                Match(bH, bPts, P, dRcut, distCut, mb, hDRB, hDistB, hDt, hTvsL, hHitB, hDtVz, vz, ref, hLdiff, hLvsEu, hDtCorr);
                Match(eH, ePts, P, dRcut, distCut, me, hDRE, hDistE, hDt, hTvsL, hHitB, hDtVz, vz, ref, hLdiff, hLvsEu, hDtCorr);

                TL all = mb; all.insert(all.end(), me.begin(), me.end());
                hNHits->Fill(all.size());
                if (all.empty()) continue;
                ++nHit;

                auto mass = [&](const TL &v) {
                    double b = CombineBeta(v);
                    if (b <= 0) return -999.0;
                    double x = p*p*(1.0/(b*b) - 1.0);
                    return x >= 0 ? sqrt(x) : -sqrt(-x);
                };
                if (!mb.empty()) { double mm = mass(mb); if (mm > -900) hMB->Fill(mm); }
                if (!me.empty()) { double mm = mass(me); if (mm > -900) hME->Fill(mm); }

                double b = CombineBeta(all);
                if (b <= 0) continue;
                ++nBeta;
                double mAll = mass(all);
                hM->Fill(mAll);
                (rcp.getCharge() > 0 ? hMPos : hMNeg)->Fill(mAll);
                hMvsP->Fill(p, mAll);
                hMp[p < 0.3 ? 0 : (p < 0.5 ? 1 : (p < 0.7 ? 2 :(p<0.9 ? 3 : (p<1.1 ? 4: 5 ))))]->Fill(mAll);
                hBetaP->Fill(p, b);
            }
        }

        cout << "==== " << s.name << " (" << nProc << " ev) ====\n"
             << "  tracks (p<pMax)          : " << nTrk << "\n"
             << "  bez rcp.getTracks()      : " << nNoTrack << "\n"
             << "  z punktami seg. w ToF    : " << nSegPts << "\n"
             << "  z dopasowanym hitem ToF  : " << nHit << "\n"
             << "  z poprawna beta          : " << nBeta << "\n"
             << "  punkty seg. z L < chord  : " << nBadPath << "  (odrzucone)" << endl;

        // nachylenie surowego (t - L/c) vs vz; oczekiwane ~ -1/c = -0.00334 ns/mm
        auto *pf = hDtVz->ProfileX(Form("dTVsVz_prof_%s", s.name.c_str()));
        pf->Fit("pol1", "Q", "", -100, 100);
        if (pf->GetFunction("pol1"))
            cout << "  slope dT/dvz = " << pf->GetFunction("pol1")->GetParameter(1)
                 << " ns/mm  (-1/c = " << -1.0 / c_light << ")" << endl;

        out.Write();
        out.Close();
    }
}