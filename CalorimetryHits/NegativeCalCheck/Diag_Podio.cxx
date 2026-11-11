// Diag_Podio.cxx  -  zrzut tego, co widzi podio dla jednego detektora (pierwsze eventy).
// Uruchom: root -l -q Diag_Podio.cxx > diag_podio.txt
// Porownaj z: Diag_Branches.cxx > diag_branches.txt   (diff / wzrokowo)

#include <podio/Frame.h>
#include <podio/ROOTReader.h>
#include <edm4eic/ClusterCollection.h>
#include <edm4eic/MCRecoClusterParticleAssociationCollection.h>
#include <edm4eic/CalorimeterHitCollection.h>
#include <iostream>
#include <string>

using namespace std;

void Diag_Podio()
{
    // --- ustaw tutaj (jeden konkretny plik, zeby kolejnosc eventow byla ta sama) ---
    string file     = "/run/media/epic/Data/Background/Pions/Continuous/reco_*.root";
    string cluster  = "HcalEndcapNClusters";
    string assoc    = "HcalEndcapNClusterAssociations";
    string filter   = "HcalEndcapN";   // ktore kolekcje wypisac (ID, rozmiar)
    int    maxPrint = 3;
    // ------------------------------------------------------------------------------

    podio::ROOTReader reader;
    reader.openFiles({file});
    unsigned nEntries = reader.getEntries("events");
    int printed = 0;

    for (unsigned e = 0; e < nEntries && printed < maxPrint; ++e)
    {
        podio::Frame frame(reader.readEntry("events", e));
        const auto &clusters = frame.get<edm4eic::ClusterCollection>(cluster);
        const auto &assocs   = frame.get<edm4eic::MCRecoClusterParticleAssociationCollection>(assoc);
        if (clusters.size() == 0) continue;
        printed++;

        cout << "=== EVENT " << e << " ===" << endl;
        for (const auto &name : frame.getAvailableCollections()) {
            if (name.find(filter) == string::npos) continue;
            auto *c = frame.get(name);
            if (c) cout << "  coll " << name << "  id=" << c->getID() << "  size=" << c->size() << endl;
        }
        cout << "  clusters=" << clusters.size() << "  assocs=" << assocs.size() << endl;

        for (size_t i = 0; i < clusters.size(); ++i) {
            const auto cl = clusters[i];
            cout << "  cluster " << i << "  E=" << cl.getEnergy() << "  nHits=" << cl.getHits().size();
            if (cl.getHits().size() > 0) {
                auto h = cl.getHits(0);
                cout << "  hit0: collID=" << h.getObjectID().collectionID
                     << " idx=" << h.getObjectID().index << " E=" << h.getEnergy()
                     << " t=" << h.getTime() << " z=" << h.getPosition().z;
            }
            cout << endl;
        }
        for (size_t j = 0; j < assocs.size(); ++j) {
            const auto a = assocs[j];
            cout << "  assoc " << j << "  rec.idx=" << a.getRec().getObjectID().index
                 << "  sim.idx=" << a.getSim().getObjectID().index
                 << "  weight=" << a.getWeight() << endl;
        }
    }
}