void run() {

  // --- LIBRARY PATHS
  gSystem->AddDynamicPath("/opt/software/linux-x86_64_v2/podio-1.6-72jyqdog5azucuny2gddj7wcvv5oanim/lib");
  gSystem->AddDynamicPath("/opt/software/linux-x86_64_v2/edm4hep-0.99.4-jzj4syggx5pyn6ycmutklcvgctaepkvh/lib");
  gSystem->AddDynamicPath("/usr/local/eic/EICrecon/lib");
  gSystem->AddDynamicPath("/opt/local/lib");

  // --- INCLUDE PATHS
  gSystem->AddIncludePath("-I/opt/software/linux-x86_64_v2/podio-1.6-72jyqdog5azucuny2gddj7wcvv5oanim/include");
  gSystem->AddIncludePath("-I/opt/software/linux-x86_64_v2/edm4hep-0.99.4-jzj4syggx5pyn6ycmutklcvgctaepkvh/include");
  gSystem->AddIncludePath("-I/usr/local/eic/EICrecon/include");
  gSystem->AddIncludePath("-I/opt/local/include");

  // --- Load libs
  gSystem->Load("libpodio.so");
  gSystem->Load("libpodioRootIO.so");

  gSystem->Load("libedm4hep.so");
  gSystem->Load("libedm4hepDict.so");

  gSystem->Load("libedm4eic.so");
  gSystem->Load("libedm4eicDict.so");

  gROOT->ProcessLine(".L PodioIDAnalysis.cxx");
  gROOT->ProcessLine("PodioIDAnalysis();");
  //gROOT->ProcessLine(".L ToFFastSim.cxx");
  //gROOT->ProcessLine("ToFFastSim();");
}