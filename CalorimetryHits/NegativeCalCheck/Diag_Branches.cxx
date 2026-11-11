// Diag_Branches.cxx  -  ten sam zrzut co Diag_Podio.cxx, ale z flat branchy.
// Uruchom: root -l -q Diag_Branches.cxx > diag_branches.txt

#include <TChain.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace std;

void Diag_Branches()
{
    // --- ustaw tutaj (te same wartosci co w Diag_Podio.cxx) ---
    //string file     = "/run/media/epic/Data/Background/Pions/Continuous/reco_*.root";
    string file     = "/run/media/epic/Data/Muons/Grape-10x275/Current/reco10x275_1*.root";
    string cluster  = "HcalEndcapNClusters";
    string assoc    = "HcalEndcapNClusterAssociations";
    vector<string> hitColls = {"HcalEndcapNRecHits", "HcalEndcapNMergedHits"};  // kandydaci
    int    maxPrint = 3;
    // ----------------------------------------------------------

    TChain chain("events");
    chain.Add(file.c_str());
    TTreeReader r(&chain);

    TTreeReaderArray<float>        cE(r, (cluster + ".energy").c_str());
    TTreeReaderArray<unsigned int> hB(r, (cluster + ".hits_begin").c_str());
    TTreeReaderArray<unsigned int> hE(r, (cluster + ".hits_end").c_str());
    TTreeReaderArray<int>          hI(r, ("_" + cluster + "_hits.index").c_str());
    TTreeReaderArray<int>          aSim(r, ("_" + assoc + "_sim.index").c_str());
    TTreeReaderArray<int>          aRec(r, ("_" + assoc + "_rec.index").c_str());
    TTreeReaderArray<float>        aW(r, (assoc + ".weight").c_str());

    // kandydaci hitow - tylko jesli branch istnieje w pliku
    vector<string> names;
    vector<unique_ptr<TTreeReaderArray<float>>> hEn, hT, hZ;
    for (auto &c : hitColls) {
        if (!chain.GetBranch((c + ".energy").c_str())) { cout << "# brak brancha " << c << ".energy" << endl; continue; }
        names.push_back(c);
        hEn.push_back(make_unique<TTreeReaderArray<float>>(r, (c + ".energy").c_str()));
        hT.push_back(make_unique<TTreeReaderArray<float>>(r, (c + ".time").c_str()));
        hZ.push_back(make_unique<TTreeReaderArray<float>>(r, (c + ".position.z").c_str()));
    }

    int ev = -1, printed = 0;
    while (r.Next() && printed < maxPrint) {
        ev++;
        if (cE.GetSize() == 0) continue;
        printed++;

        cout << "=== EVENT " << ev << " ===" << endl;
        for (size_t k = 0; k < names.size(); ++k)
            cout << "  coll " << names[k] << "  size=" << hEn[k]->GetSize() << endl;
        cout << "  clusters=" << cE.GetSize() << "  assocs=" << aSim.GetSize() << endl;

        for (size_t i = 0; i < cE.GetSize(); ++i) {
            unsigned nH = hE[i] - hB[i];
            cout << "  cluster " << i << "  E=" << cE[i] << "  nHits=" << nH;
            if (nH > 0) {
                int idx = hI[hB[i]];
                cout << "  hit0 idx=" << idx;
                for (size_t k = 0; k < names.size(); ++k) {
                    cout << "  [" << names[k] << ": ";
                    if (idx >= 0 && (size_t)idx < hEn[k]->GetSize())
                        cout << "E=" << (*hEn[k])[idx] << " t=" << (*hT[k])[idx] << " z=" << (*hZ[k])[idx];
                    else cout << "OUT OF RANGE";
                    cout << "]";
                }
            }
            cout << endl;
        }
        for (size_t j = 0; j < aSim.GetSize(); ++j)
            cout << "  assoc " << j << "  rec.idx=" << aRec[j] << "  sim.idx=" << aSim[j]
                 << "  weight=" << aW[j] << endl;
    }
}