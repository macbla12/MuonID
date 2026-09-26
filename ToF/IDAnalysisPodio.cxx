// IDAnalysis_EDM4eic.cxx -- ToF section following the MuonID.cxx approach
//
// Difference from the original IDAnalysis.cxx: instead of four separate
// TTreeReaderArray<float> branches (PosX/PosY/PosZ/Time), read hits as
// edm4eic::TrackerHit objects from podio::Frame (event->get<...>),
// as in MuonID.cxx. Matching logic (dR, dist_cut, ToFSim, CombineBeta) is
// unchanged; only data access is different.
#include <glob.h>

#include <edm4eic/TrackerHitCollection.h>
#include <edm4eic/ReconstructedParticleCollection.h>
#include <podio/ROOTReader.h>
#include <podio/Frame.h>
#include <TLorentzVector.h>
#include <TVector3.h>
#include <TVector2.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TMath.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TFile.h>
#include <TGraph.h>
#include <vector>
#include <utility>
#include <cmath>
#include <string>

#include "ToFSim.cxx" // ToFSim(), ToFResults, c_light

// =====================================================================
// Keep constants, structures, and helper functions at file scope, following
// MuonID_Detail in MuonID.cxx, so IDAnalysisPodio() only invokes them below.
// C++ does not allow defining free functions or templates inside another function.
// =====================================================================
namespace IDAnalysis_Detail {

// Cuts: same values as the original, now defined as constants (as in MuonID).
static constexpr double DR_CUT_BARREL   = 0.8;
static constexpr double DR_CUT_ENDCAP   = 0.8;
static constexpr double DIST_CUT_BARREL = 6.0; // mm
static constexpr double DIST_CUT_ENDCAP = 6.0; // mm

struct BetaEstimate { double beta; bool valid; };

static BetaEstimate CombineBeta(const std::vector<std::pair<double, double>>& hits)
{
    double sumXX = 0.0, sumXT = 0.0;
    for (auto& h : hits)
    {
        double t = h.first;
        double L = h.second;
        double x = L / c_light;
        sumXX += x * x;
        sumXT += x * t;
    }
    if (sumXX <= 0.0) return {0.0, false};
    double u_hat = sumXT / sumXX;
    if (u_hat <= 0.0) return {0.0, false};
    return {1.0 / u_hat, true};
}

// ToF match result for one track, corresponding to the local variables
// originally used in the IDAnalysis particle loop.
struct TrackToFMatch
{
    std::vector<std::pair<double, double>> matchedBarrel; // {time, length}
    std::vector<std::pair<double, double>> matchedEndcap;
};

// Match ToF hits to one track, following the MuonID approach:
// - use edm4eic::TrackerHitCollection collections instead of TTreeReaderArray
// - iterate over hit objects with a range-based loop (hit.getPosition(), hit.getTime())
// - keep diagnostic histograms optional (nullptr disables them); this analysis
//   macro uses them, while MuonID does not.
template <typename HitCollection>
TrackToFMatch MatchToFHits(const TLorentzVector& Partic, int charge,
                            const HitCollection& barrelHits,
                            const HitCollection& endcapHits,
                            TH1D* hEtaBarrel = nullptr, TH1D* hPhiBarrel = nullptr,
                            TH1D* hTimeBarrel = nullptr,
                            TH1D* hDEtaBarrel = nullptr, TH1D* hDPhiBarrel = nullptr,
                            TH1D* hDRBarrel = nullptr, TH1D* hDistBarrel = nullptr,
                            TH1D* hEtaEndcap = nullptr, TH1D* hPhiEndcap = nullptr,
                            TH1D* hTimeEndcap = nullptr,
                            TH1D* hDEtaEndcap = nullptr, TH1D* hDPhiEndcap = nullptr,
                            TH1D* hDREndcap = nullptr, TH1D* hDistEndcap = nullptr,
                            TH1D* hPathLenMinusChord = nullptr,
                            TH1D* hToFSimFailedFrac = nullptr)
{
    TrackToFMatch result;

    const double trackEta = Partic.Eta();
    const double trackPhi = Partic.Phi();
    static constexpr double DEG = 180.0 / TMath::Pi();

    // --- BARREL: petla po obiektach hit z kolekcji edm4eic, nie po indeksie ---
    for (const auto& hit : barrelHits)
    {
        auto pos = hit.getPosition();
        TVector3 ToFPos(pos.x, pos.y, pos.z);

        if (hEtaBarrel)  hEtaBarrel->Fill(ToFPos.Eta());
        if (hPhiBarrel)  hPhiBarrel->Fill(ToFPos.Phi() * DEG);
        if (hTimeBarrel) hTimeBarrel->Fill(hit.getTime());

        double dEta = trackEta - ToFPos.Eta();
        double dPhi = TVector2::Phi_mpi_pi(trackPhi - ToFPos.Phi());
        double dR   = std::sqrt(dEta * dEta + dPhi * dPhi);

        if (hDEtaBarrel) hDEtaBarrel->Fill(dEta);
        if (hDPhiBarrel) hDPhiBarrel->Fill(dPhi);
        if (hDRBarrel)   hDRBarrel->Fill(dR);

        if (dR < DR_CUT_BARREL && charge * dPhi < 0)
        {
            ToFResults tof = ToFSim(Partic, charge, ToFPos);

            if (hToFSimFailedFrac) hToFSimFailedFrac->Fill(tof.DistanceCheck ? 0 : 1);
            if (hDistBarrel)       hDistBarrel->Fill(tof.distance_to_TOF);

            if (tof.distance_to_TOF < DIST_CUT_BARREL)
            {
                double length = tof.DistanceCheck ? tof.track_length : ToFPos.Mag();
                if (tof.DistanceCheck && hPathLenMinusChord)
                    hPathLenMinusChord->Fill(tof.track_length - ToFPos.Mag());

                result.matchedBarrel.push_back({hit.getTime(), length});
            }
        }
    }

    // --- ENDCAP: analogicznie ---
    for (const auto& hit : endcapHits)
    {
        auto pos = hit.getPosition();
        TVector3 ToFPos(pos.x, pos.y, pos.z);

        if (hEtaEndcap)  hEtaEndcap->Fill(ToFPos.Eta());
        if (hPhiEndcap)  hPhiEndcap->Fill(ToFPos.Phi() * DEG);
        if (hTimeEndcap) hTimeEndcap->Fill(hit.getTime());

        double dEta = trackEta - ToFPos.Eta();
        double dPhi = TVector2::Phi_mpi_pi(trackPhi - ToFPos.Phi());
        double dR   = std::sqrt(dEta * dEta + dPhi * dPhi);

        if (hDEtaEndcap) hDEtaEndcap->Fill(dEta);
        if (hDPhiEndcap) hDPhiEndcap->Fill(dPhi);
        if (hDREndcap)   hDREndcap->Fill(dR);

        if (dR < DR_CUT_ENDCAP && charge * dPhi < 0)
        {
            ToFResults tof = ToFSim(Partic, charge, ToFPos);

            if (hToFSimFailedFrac) hToFSimFailedFrac->Fill(tof.DistanceCheck ? 0 : 1);
            if (hDistEndcap)       hDistEndcap->Fill(tof.distance_to_TOF);

            // NOTE: the original IDAnalysis.cxx copied the barrel cut here
            // (dist_cut_barrel instead of dist_cut_endcap), which appears to
            // be a bug. This uses DIST_CUT_ENDCAP; to reproduce the original
            // behavior exactly, switch back to DIST_CUT_BARREL.
            if (tof.distance_to_TOF < DIST_CUT_ENDCAP)
            {
                double length = tof.DistanceCheck ? tof.track_length : ToFPos.Mag();
                if (tof.DistanceCheck && hPathLenMinusChord)
                    hPathLenMinusChord->Fill(tof.track_length - ToFPos.Mag());

                result.matchedEndcap.push_back({hit.getTime(), length});
            }
        }
    }

    return result;
}

} // namespace IDAnalysis_Detail

std::vector<std::string> ExpandGlob(const std::string &pattern)
{
    std::vector<std::string> filenames;
    glob_t glob_result;
    glob(pattern.c_str(), GLOB_TILDE, nullptr, &glob_result);
    for (size_t i = 0; i < glob_result.gl_pathc; ++i)
        filenames.push_back(std::string(glob_result.gl_pathv[i]));
    globfree(&glob_result);
    return filenames;
}

// =====================================================================
// IDAnalysisPodio() -- follows the example.cxx and MuonID.cxx approach.
// Read events from podio::Frame, retrieve collections through ROOTReader,
// and work directly with edm4eic::ReconstructedParticleCollection and ToF hits.
// =====================================================================
void IDAnalysisPodio()
{
    using namespace IDAnalysis_Detail;

    static double MuonMass     = 0.1056583;
    static double PionMass     = 0.13957039;
    static double ElectronMass = 0.00051099895;
    static double KaonMass     = 0.493677;
    static double ProtonMass   = 0.93827208;

    std::vector<double> masses = {MuonMass, ElectronMass, PionMass, KaonMass, ProtonMass};
    std::vector<int>    pdgs   = {13, 11, 211, 321, 2212};
    std::vector<double> masses2(masses.size());
    for (size_t i = 0; i < masses.size(); ++i) masses2[i] = masses[i] * masses[i];

    gROOT->SetBatch(kTRUE);
    gROOT->ProcessLine("gErrorIgnoreLevel = 3000;");

    const double DEG = 180.0 / TMath::Pi();
    TFile* PaperFile = new TFile("Plots/ToFPlots.root", "RECREATE");

    static constexpr int NumOfFiles = 1;
    static constexpr int nPbins = 25;

    double totalTracks[NumOfFiles][nPbins] = {0};
    double foundTracks[NumOfFiles][nPbins] = {0};
    double goodTracksP[NumOfFiles][nPbins] = {0};
    double badTracksP[NumOfFiles][nPbins]  = {0};

    TH1D *BToFEtaRangeHist[NumOfFiles], *BToFPhiRangeHist[NumOfFiles];
    TH1D *EToFEtaRangeHist[NumOfFiles], *EToFPhiRangeHist[NumOfFiles];
    TH1D *ToFBTimeHist[NumOfFiles], *ToFETimeHist[NumOfFiles];
    TH1D *NumberOfBHits[NumOfFiles], *NumberOfEHits[NumOfFiles];
    TH1D *EffVsP[NumOfFiles];
    TH1D *PurityVsP[NumOfFiles];
    TH2D *BetaVsMom[NumOfFiles];
    TH1D *MassSqHist[NumOfFiles];
    TH2D *MassSqVsMom[NumOfFiles];
    TH1D *dEtaBarrel[NumOfFiles], *dPhiBarrel[NumOfFiles], *dRBarrel[NumOfFiles], *DistanceToFBarrel[NumOfFiles];
    TH1D *dEtaEndcap[NumOfFiles], *dPhiEndcap[NumOfFiles], *dREndcap[NumOfFiles], *DistanceToFEndcap[NumOfFiles];
    TH1D *NMatchedHitsBarrel[NumOfFiles], *NMatchedHitsEndcap[NumOfFiles];
    TH1D *PathLenMinusChord[NumOfFiles];
    TH1D *ToFSimFailedFrac[NumOfFiles];

    const std::string pattern = "/run/media/epic/Data/Background/Muons/Continuous/reco_aMuonsLow*.root";
    //const std::string pattern ="/run/media/epic/Data/Muons/Grape-10x275/Current/reco10x275_1*.root";

    std::vector<std::string> inputFiles = ExpandGlob(pattern);
    if (inputFiles.empty()) {
        std::cerr << "No files found matching pattern: " << pattern << std::endl;
        return;
    }

    const int File = 0;
    const std::string name = "Muons";

    BToFEtaRangeHist[File] = new TH1D(Form("BToFEtaRangeHist%s", name.c_str()),
                                     Form("BToFEtaRangeHist%s", name.c_str()),
                                     50, -2.5, 2.5);
    BToFPhiRangeHist[File] = new TH1D(Form("BToFPhiRangeHist%s", name.c_str()),
                                     Form("BToFPhiRangeHist%s", name.c_str()),
                                     50, -180, 180);

    EToFEtaRangeHist[File] = new TH1D(Form("EToFEtaRangeHist%s", name.c_str()),
                                     Form("EToFEtaRangeHist%s", name.c_str()),
                                     50, 1.5, 3.5);
    EToFPhiRangeHist[File] = new TH1D(Form("EToFPhiRangeHist%s", name.c_str()),
                                     Form("EToFPhiRangeHist%s", name.c_str()),
                                     50, -180, 180);

    ToFBTimeHist[File] = new TH1D(Form("ToFBarrelTimeHist%s", name.c_str()),
                                 Form("ToFBarrelTimeHist%s", name.c_str()),
                                 50, 0, 15);
    ToFETimeHist[File] = new TH1D(Form("ToFEndcapTimeHist%s", name.c_str()),
                                 Form("ToFEndcapTimeHist%s", name.c_str()),
                                 50, 0, 15);

    NumberOfBHits[File] = new TH1D(Form("NumberOfBarrelHits%s", name.c_str()),
                                  Form("NumberOfBarrelHits%s", name.c_str()),
                                  10, -0.5, 9.5);
    NumberOfEHits[File] = new TH1D(Form("NumberOfEndcapHits%s", name.c_str()),
                                  Form("NumberOfEndcapHits%s", name.c_str()),
                                  10, -0.5, 9.5);

    EffVsP[File] = new TH1D(Form("EffVsP_%s", name.c_str()),
                           Form("Efficiency vs P %s", name.c_str()),
                           nPbins, 0.0, 2.5);
    PurityVsP[File] = new TH1D(Form("PurityVsP_%s", name.c_str()),
                              Form("Purity vs P %s", name.c_str()),
                              nPbins, 0.0, 2.5);

    BetaVsMom[File] = new TH2D(Form("BetaVsMom_%s", name.c_str()),
                              Form("BetaVsMom_%s (combined per-track);p [GeV];#beta", name.c_str()),
                              100, 0, 2, 100, 0.7, 1.1);

    MassSqHist[File] = new TH1D(Form("MassHist_%s", name.c_str()),
                               Form("Mass (combined per-track) %s;m [GeV];Counts", name.c_str()),
                               200, -0.05, 0.35);
    MassSqVsMom[File] = new TH2D(Form("MassSqVsMom_%s", name.c_str()),
                                Form("MassSqVsMom_%s;p [GeV];m^{2} [GeV^{2}]", name.c_str()),
                                100, 0, 2, 150, -0.1, 1.1);

    dEtaBarrel[File] = new TH1D(Form("dEtaBarrel_%s", name.c_str()),
                                Form("dEta (track-hit) Barrel %s;#Delta#eta;Counts", name.c_str()),
                                200, -1.0, 1.0);
    dPhiBarrel[File] = new TH1D(Form("dPhiBarrel_%s", name.c_str()),
                                Form("dPhi (track-hit) Barrel %s;#Delta#phi [rad];Counts", name.c_str()),
                                200, -1.0, 1.0);
    dRBarrel[File] = new TH1D(Form("dRBarrel_%s", name.c_str()),
                              Form("dR (track-hit) Barrel %s;#DeltaR;Counts", name.c_str()),
                              200, 0.0, 2.0);
    DistanceToFBarrel[File] = new TH1D(Form("DistanceToFBarrel_%s", name.c_str()),
                                      Form("DistanceToFBarrel (track-hit) Barrel %s;DistanceToFBarrel [mm];Counts", name.c_str()),
                                      200, 0.0, 100.0);

    dEtaEndcap[File] = new TH1D(Form("dEtaEndcap_%s", name.c_str()),
                                Form("dEta (track-hit) Endcap %s;#Delta#eta;Counts", name.c_str()),
                                200, -1.0, 1.0);
    dPhiEndcap[File] = new TH1D(Form("dPhiEndcap_%s", name.c_str()),
                                Form("dPhi (track-hit) Endcap %s;#Delta#phi [rad];Counts", name.c_str()),
                                200, -1.0, 1.0);
    dREndcap[File] = new TH1D(Form("dREndcap_%s", name.c_str()),
                              Form("dR (track-hit) Endcap %s;#DeltaR;Counts", name.c_str()),
                              200, 0.0, 2.0);
    DistanceToFEndcap[File] = new TH1D(Form("DistanceToFEndcap_%s", name.c_str()),
                                      Form("DistanceToFEndcap (track-hit) Endcap %s;DistanceToFEndcap [mm];Counts", name.c_str()),
                                      200, 0.0, 100.0);

    NMatchedHitsBarrel[File] = new TH1D(Form("NMatchedHitsBarrel_%s", name.c_str()),
                                       Form("Matched hits per track (Barrel) %s;N hits;Tracks", name.c_str()),
                                       10, -0.5, 9.5);
    NMatchedHitsEndcap[File] = new TH1D(Form("NMatchedHitsEndcap_%s", name.c_str()),
                                       Form("Matched hits per track (Endcap) %s;N hits;Tracks", name.c_str()),
                                       10, -0.5, 9.5);

    PathLenMinusChord[File] = new TH1D(Form("PathLenMinusChord_%s", name.c_str()),
                                      Form("RK4 path length minus straight-line chord %s;#Delta L [mm];Counts", name.c_str()),
                                      200, -5.0, 50.0);

    ToFSimFailedFrac[File] = new TH1D(Form("ToFSimFailedFrac_%s", name.c_str()),
                                     Form("ToFSim DistanceCheck failed (1) vs ok (0) %s", name.c_str()),
                                     2, -0.5, 1.5);

    podio::ROOTReader reader;
    reader.openFiles(inputFiles);

    const long long nEvents = 100000; //reader.getEntries("events");
    double FoundParticles = 0;
    double particscount = 0;
    double goodcount = 0;
    double badcount = 0;

    for (long long iev = 0; iev < nEvents; ++iev)
    {
        const auto event = podio::Frame(reader.readNextEntry("events"));

        const auto& reco_parts = event.get<edm4eic::ReconstructedParticleCollection>("ReconstructedChargedParticles");
        const auto& tofBarrelHits = event.get<edm4eic::TrackerHitCollection>("TOFBarrelRecHits");
        const auto& tofEndcapHits = event.get<edm4eic::TrackerHitCollection>("TOFEndcapRecHits");

        for (const auto& rcp : reco_parts)
        {
            const auto mom = rcp.getMomentum();
            TLorentzVector partic;
            partic.SetPxPyPzE(mom.x, mom.y, mom.z, rcp.getEnergy());

            if (partic.P() > 1.0) continue;

            particscount++;

            const double p = partic.P();
            const int pbin = static_cast<int>(p / 0.1);
            if (pbin >= 0 && pbin < nPbins) totalTracks[File][pbin]++;

            const int charge = static_cast<int>(rcp.getCharge());

            auto match = MatchToFHits(partic, charge, tofBarrelHits, tofEndcapHits,
                                     BToFEtaRangeHist[File], BToFPhiRangeHist[File],
                                     ToFBTimeHist[File], dEtaBarrel[File], dPhiBarrel[File],
                                     dRBarrel[File], DistanceToFBarrel[File],
                                     EToFEtaRangeHist[File], EToFPhiRangeHist[File],
                                     ToFETimeHist[File], dEtaEndcap[File], dPhiEndcap[File],
                                     dREndcap[File], DistanceToFEndcap[File],
                                     PathLenMinusChord[File], ToFSimFailedFrac[File]);

            NumberOfBHits[File]->Fill((double)match.matchedBarrel.size());
            NMatchedHitsBarrel[File]->Fill((double)match.matchedBarrel.size());
            NumberOfEHits[File]->Fill((double)match.matchedEndcap.size());
            NMatchedHitsEndcap[File]->Fill((double)match.matchedEndcap.size());

            const bool found = !match.matchedBarrel.empty() || !match.matchedEndcap.empty();
            if (found) FoundParticles++;
            if (!found) continue;

            std::vector<std::pair<double, double>> allMatched;
            allMatched.insert(allMatched.end(), match.matchedBarrel.begin(), match.matchedBarrel.end());
            allMatched.insert(allMatched.end(), match.matchedEndcap.begin(), match.matchedEndcap.end());

            const auto be = CombineBeta(allMatched);
            if (!be.valid) continue;

            BetaVsMom[File]->Fill(p, be.beta);

            const double msq = p * p * (1.0 / (be.beta * be.beta) - 1.0);
            MassSqHist[File]->Fill(std::sqrt(msq));
            MassSqVsMom[File]->Fill(p, msq);

            int bestPDG = 0;
            double bestDiff = 1e18;
            for (size_t i = 0; i < pdgs.size(); ++i)
            {
                const double diff = std::fabs(msq - masses2[i]);
                if (diff < bestDiff)
                {
                    bestDiff = diff;
                    bestPDG = pdgs[i];
                }
            }

            if (pbin >= 0 && pbin < nPbins)
            {
                foundTracks[File][pbin]++;
                if (bestPDG == 13) goodTracksP[File][pbin]++;
                else badTracksP[File][pbin]++;
            }

            if (bestPDG == 13) goodcount++;
            else badcount++;
        }
    }

    std::cout << "===========================" << std::endl;
    std::cout << "End of " << name << " file" << std::endl;
    std::cout << "Number of events: " << nEvents << std::endl;
    std::cout << "Found particles: " << FoundParticles << "   All particles: " << particscount << std::endl;
    std::cout << "Found Ratio: " << (particscount > 0 ? FoundParticles * 100.0 / particscount : 0.0) << '%' << std::endl;
    std::cout << "Purity: " << (goodcount + badcount > 0 ? goodcount * 100.0 / (goodcount + badcount) : 0.0) << '%' << std::endl;
    std::cout << "===========================" << std::endl;

    for (int ib = 0; ib < nPbins; ++ib)
    {
        const double eff = (totalTracks[File][ib] > 0.0)
            ? foundTracks[File][ib] / totalTracks[File][ib]
            : 0.0;
        const double purity = (goodTracksP[File][ib] + badTracksP[File][ib] > 0.0)
            ? goodTracksP[File][ib] / (goodTracksP[File][ib] + badTracksP[File][ib])
            : 0.0;

        EffVsP[File]->SetBinContent(ib + 1, eff);
        PurityVsP[File]->SetBinContent(ib + 1, purity);
    }

    BToFEtaRangeHist[File]->Write();
    BToFPhiRangeHist[File]->Write();
    EToFEtaRangeHist[File]->Write();
    EToFPhiRangeHist[File]->Write();
    ToFBTimeHist[File]->Write();
    ToFETimeHist[File]->Write();
    NumberOfBHits[File]->Write();
    NumberOfEHits[File]->Write();
    EffVsP[File]->Write();
    PurityVsP[File]->Write();
    BetaVsMom[File]->Write();
    MassSqHist[File]->Write();
    MassSqVsMom[File]->Write();
    dEtaBarrel[File]->Write();
    dPhiBarrel[File]->Write();
    dRBarrel[File]->Write();
    DistanceToFBarrel[File]->Write();
    dEtaEndcap[File]->Write();
    dPhiEndcap[File]->Write();
    dREndcap[File]->Write();
    DistanceToFEndcap[File]->Write();
    NMatchedHitsBarrel[File]->Write();
    NMatchedHitsEndcap[File]->Write();
    PathLenMinusChord[File]->Write();
    ToFSimFailedFrac[File]->Write();

    const int nCurvePoints = 200;
    TGraph* betaMuon = new TGraph(nCurvePoints);
    TGraph* betaElec = new TGraph(nCurvePoints);
    TGraph* betaPion = new TGraph(nCurvePoints);
    TGraph* betaKaon = new TGraph(nCurvePoints);
    TGraph* betaProton = new TGraph(nCurvePoints);

    for (int i = 0; i < nCurvePoints; ++i)
    {
        const double p = 5.0 * i / (nCurvePoints - 1);
        auto beta = [&](double m) { const double E = std::sqrt(p * p + m * m); return p / E; };
        betaMuon  ->SetPoint(i, p, beta(MuonMass));
        betaElec  ->SetPoint(i, p, beta(ElectronMass));
        betaPion  ->SetPoint(i, p, beta(PionMass));
        betaKaon  ->SetPoint(i, p, beta(KaonMass));
        betaProton->SetPoint(i, p, beta(ProtonMass));
    }

    betaMuon  ->SetName("BetaCurve_Muon");
    betaElec  ->SetName("BetaCurve_Electron");
    betaPion  ->SetName("BetaCurve_Pion");
    betaKaon  ->SetName("BetaCurve_Kaon");
    betaProton->SetName("BetaCurve_Proton");
    betaMuon  ->Write();
    betaElec  ->Write();
    betaPion  ->Write();
    betaKaon  ->Write();
    betaProton->Write();

    PaperFile->Close();
}