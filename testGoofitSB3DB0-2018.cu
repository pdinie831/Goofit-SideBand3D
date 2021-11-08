//
//
// 
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sys/stat.h>
#include "Riostream.h"
#include <map>
#include <string>
//#include <boost/algorithm/string.hpp>
//#include <boost/algorithm/string/trim.hpp>f
#include <vector>
#include <math.h>
//#include <TCint.h>
#include <TGenericClassInfo.h> 
#include <TMath.h>
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TH3F.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TH3D.h>
#include <TF1.h>
#include <TF2.h>
#include <TF3.h>
#include <TMatrixD.h>
#include <TVirtualFitter.h>
#include <TROOT.h>
#include <TEnv.h>
#include <TSystem.h>
#include <TTree.h>
#include "TBranch.h"
#include <TApplication.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TStyle.h> 
#include <TF1.h>  
#include <TF2.h> 
#include <TLorentzVector.h>
#include <TVector3.h>
#include "TDSet.h"
#include "TChain.h"
#include <time.h> 
#include <TSystemDirectory.h>
#include <TEfficiency.h>
#include <TLegend.h>
#include <TMinuit.h>
#include "Math/WrappedMultiTF1.h"
#include "TRandom.h" 
#include "TRandom3.h" 
#include  <TStopwatch.h>
#include "TH1F.h"
#include "TH2F.h"			// unused?
#include "TStyle.h"
#include "TCanvas.h"
#include <TGraphAsymmErrors.h>
#include <TFrame.h>
#include <TFitResult.h>
#include <TFitter.h>
#include "Fit/Fitter.h"
#include <TMatrixDSym.h>
#include <TBinomialEfficiencyFitter.h>
#include <TKDTreeBinning.h>
#include <TH2Poly.h>
//#if !defined(__CINT__) || defined(__MAKECINT__)
#include <RooFit.h>
#include <RooMinuit.h>
#include <RooRealVar.h>
#include <RooConstVar.h>
#include <RooPlot.h>
#include <RooDataSet.h>
#include <RooDataHist.h>
#include <RooHistPdf.h>
#include <RooRealVar.h>
#include <RooAbsReal.h>
#include <RooArgSet.h>
#include <RooRealProxy.h>
#include <RooCategoryProxy.h>
#include <RooAbsCategory.h>
#include <RooProdPdf.h>
#include <RooAddPdf.h>
#include <RooProduct.h>
#include <RooGaussian.h>
#include <RooCBShape.h>
#include <RooBifurGauss.h>
#include <RooPolynomial.h>
#include <RooChebychev.h>
#include <RooWorkspace.h>
#include <RooExponential.h>
#include <RooBernstein.h>
#include <RooErrorVar.h>
#include <RooFitResult.h>
#include <RooRangeBinning.h>
#include <RooBinning.h>
#include <TRatioPlot.h>
#include "GBRMath.h"
#include "RooDoubleCBFast.h"
#include "RooBernsteinSideband.h"
#include <sys/time.h>
#include <sys/times.h>
#include <iostream>
//#endif
// GooFit stuff
#include <goofit/Application.h>
#include <goofit/FitManager.h>
#include <goofit/PDFs/GooPdf.h>
#include <goofit/PDFs/basic/GaussianPdf.h>
#include <goofit/PDFs/basic/PolynomialPdf.h>
#include <goofit/PDFs/combine/ProdPdf.h>
#include <goofit/PDFs/combine/ConvolutionPdf.h>
#include <goofit/PDFs/combine/EventWeightedAddPdf.h>
#include <goofit/PDFs/combine/MappedPdf.h>
#include <goofit/PDFs/combine/AddPdf.h>
#include <goofit/UnbinnedDataSet.h>
#include <goofit/fitting/FitManagerMinuit1.h>
#include <goofit/Variable.h>
//#include <goofit/PDFs/mypdf/EffiBernsteinPdf.h>
//#include <goofit/PDFs/mypdf/EffiTestBernsteinPdf.h>
//#include <goofit/PDFs/mypdf/EffiPolynomialPdf.h>
//#include <goofit/PDFs/mypdf/BernsteinPdf.h>
#include <goofit/PDFs/mypdf/RGaussianPdf.h>
#include <goofit/PDFs/mypdf/SimpleCheby2Pdf.h>
#include <goofit/PDFs/mypdf/ErfcMassPdf.h>
#include <goofit/PDFs/mypdf/BernsteinTestPdf.h>
#include <goofit/PDFs/mypdf/FastBernsteinPdf.h>
//#include <goofit/PDFs/mypdf/NormProdEffiPdf.h>
//#include <goofit/PDFs/mypdf/NormProdEffiTestPdf.h>
//#include "ExpGausPEEPdf.h" 
//#include "ExpGausMPdf.h" 
//#include "ExpGausWithIntPdf.h"
//#include "ExpGausPEEfixSigmaPdf.h" 
//#include "ExpGausProdBPdf.h"
//#include "ExpGausProdEffiBPdf.h"
//#include "ExpGausPEESigmaBPdf.h" 
//#include "PolyEffiPdf.h" 
//#include "ErfcPolyPdf.h"
//#include "ErfcMassPdf.h"
//#include "SigmoidB0Pdf.h"
//#include <goofit/PDFs/mypdf/SigmoidB0Pdf.h>
//#include "ErfEffiBpPdf.h"
//#include "SigmoidGausPdf.h"
//#include "GooFit/BivarGaussianConstrPdf.h"
//#include "GooFit/TrivarGaussianConstrPdf.h"
// System stuff
#include <fstream> 
#include <sys/time.h>
#include <sys/times.h>

timeval startTime, stopTime, totalTime;
clock_t startCPU, stopCPU; 
tms startProc, stopProc; 

using namespace std; 
using namespace GooFit;
using namespace ROOT;
using namespace RooFit;

void FitSBModel();
void CreateInputHistoFile();
int RunEra=2018;
double FitMassSpectrum(UnbinnedDataSet* dataMass, TCanvas* c2, TH1D* masHist, TH1D* pdfHist, TH1D*sigHist, TH1D* bckHist, int MaxDegree);
double FitMassSpectrumRoofit(RooDataSet* RooDataMass, TCanvas* c2, TH1D* masHist, TH1D* pdfHist, TH1D*sigHist, TH1D* bkgHist, int MaxDegreeBckg);
RooGaussian* _constrainVar(RooRealVar *var,RooWorkspace *w=0);
float*  _getFittedVar(const char* varName,RooWorkspace *w=0);
void replaceAll(std::string& str, const std::string& from, const std::string& to) ;
void replaceChar(char * txt, const char * txt1, const char * txt2) ;

//R__LOAD_LIBRARY(libRooBernsteinSideband)

//class RooBernsteinSideband;

bool minuit1;
bool wrongTagged = false;
bool SetMinuit2  = false;
bool Folded      = false;
bool integral    = false;

  char RecoDir[100]                 =  "~/p5prime/[RunEra]/skims/newphi";// RunEra  will be set after...
  char InputRecoB0TreeName[10]	    = "ntuple";
  char OutputRecoB0TreeName[10]	    = "ntuple";
  char InputFileNameRecoB0[300]     = "[RunEra]Data_All_finalSelection.root";// RunEra  will be set after...
  char ListParName[400] 	    =  "ListParValues-[RunEra]-Q2Bin-2-Bins-.txt";
  char ListPloName[400] 	    =  "ListParValues-[RunEra]-Q2Bin-2-Bins-.plo";
  char ListParNorm[410] 	    =  "ListParValues-[RunEra]-Q2Bin-2-Bins-.txt_norm";
  char ListPloNorm[410] 	    =  "ListParValues-[RunEra]-Q2Bin-2-Bins-.plo_norm";
  char FitStraName[400] 	    =  "namelist-[RunEra]-Q2Bin-2-Bins-.stra";
  char OutFileName[400] 	    =  "testGoofitSB3DB0-[RunEra]-[RunEra]-Q2Bin-1.root";
  char OutFileNameInputHisto[300]   =  "testGoofitSB3DB0-[RunEra]-[RunEra]-InputHisto-Q2Bin-1.root";
  char OutSaveFileName[400]	    =  "";
  char PDFNameRecoHisto[350]	    =  "B0-RecoHist-[RunEra]-Q2Bin-1.pdf";
  char PDFNameGeneHisto[350]	    =  "B0-GeneHist-[RunEra]-Q2Bin-1.pdf";
  char PNGNameMassHist[350]	    =  "B0-MassCheck-[RunEra]-Q2Bin-1.png";
  char PNGNameMassCheck[350]	    =  "B0-MassHist-[RunEra]-Q2Bin-1.png";
  char PNGNameMassQ2Hist[350]	    =  "B0-MassHist-[RunEra]-Q2Bin-1.png";
  char PNGNameProjXYHist[350]	    =  "";
  char PNGNameProjZYHist[350]	    =  "";
  char PNGNameProjZXHist[350]	    =  "";
  char ProjectTXT[300]		    =  "";
  char SigmaMethodTXT[100]	    =  "";
  char TaggedVarTXT[100]	    =  "";
  char FoldedTXT[100]		    =  "";
  char fitMassFileName[300]         =  "~/p5prime/massFits/results_fits_[RunEra]_fM_newbdt.root";// RunEra  will be set after...
  char fitMassFileNameJpsi[300]     =  "~/p5prime/massFits/results_fits_[RunEra]_fM_Jpsi_newbdt.root";// RunEra  will be set after...
  char fitMassFileNamePsi[300]      =  "~/p5prime/massFits/results_fits_[RunEra]_fM_Psi_newbdt.root";// RunEra  will be set after...
  char fitMassFileNameQ2Bin7[300]   =  "~/p5prime/massFits/results_fits_[RunEra]_fM_newbin7.root";// RunEra  will be set after...
  char FMTNSigma1L[10]		    ="";
  char FMTNSigma2L[10]		    ="";
  char FMTNSigma1R[10]		    ="";
  char FMTNSigma2R[10]		    ="";
//  char fitMassFileName[100]         =  "results_fits_[RunEra].root";
//  char fitMassFileName[100]         =  "rf607_fitresult.root";

  
  TFile*OutFile = 0;

 

  char PDFNameMass[350]          = "B0-Mass-[RunEra]-global.pdf";
  char PDFNameFitSB3D[350]       = "B0-FitSB3DMass-[RunEra]-global.pdf";

  char PNGNameFitSB3D[350]       = "B0-FitSB3D-[RunEra]-global.png";
  char PNGNameFitSB3DMass[350]   = "B0-FitSB3DMass-[RunEra]-global.png";
  char PNGNameFitSB3DProjX[350]  = "B0-FitSB3DProjX-[RunEra]-global.png";
  char PNGNameFitSB3DProjY[350]  = "B0-FitSB3DProjY-[RunEra]-global.png";
  char PNGNameFitSB3DProjZ[350]  = "B0-FitSB3DProjZ-[RunEra]-global.png";
  char testo[300]     = "" ;
  float MarkerSizeSet = 0.35;
  int   PlotLineWidth = 1.;

//============================
// maxDegree START
// now defined in NAMELIST 
//============================
  int maxDegree1 =0;
  int maxDegree2 =0;
  int maxDegree3 =0;
// Mass Spectrum Bernstein   
  int maxDegree  =0;
  
  int fixParam   =1;
//============================
// now defined in NAMELIST 
// maxDegree END
//============================


//
// Il Bin in q^2 !!!!
//   
  double Q2Min = 0.; 
  double Q2Max = 0.; 
  int    Q2Bin = -1;
//=================  
// Number of Normalization Integrals
//=================
  int    NormInteg = 11;
//=================
//=================
  int SETNumBinsX=100;
  int SETNumBinsY=100;
  int SETNumBinsZ=100;
//=================
//=================
  double SetMinRatio=0;
  double SetMaxRatio=3;
  double SetMinProj=0;
//   
//=================
//=================
// NFact!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  int   NFact = 4; 
// NFact!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//=================
//=================
//     MCToys
//=================

  int   NFactGen = 1;

//===================================================
//=====significance cut definition for params  ======
//=====> should be redefined for each Q2Bin!  <======
//===================================================
       double CutSignificance   = 2;
//===================================================
//===================================================
  int SigmaProbSign=1;
//  
  int FitPrintLevel=1;
  bool boolHesse = true;
  bool AutoFixPar= false;
  bool FirstFit  = false;
  int  iFitLoop  = 3;
  double xCoeffNorm =0.0;
  int    xCoeffIndex=-1;
//=================
//=================
//=================
//=================
  double ParMin =  0.;
  double ParMax =  10000;
  double RndMin =  0.01;
  double RndMax =  0.1;
//=================
//=================
//=================
//=================

  TCanvas *csignstudy=0;
  double  tagged_mass_rangeMin=5.0;
  double  tagged_mass_rangeMax=5.6;

  double XMinSign = 4.9;
  double XMaxSign = 5.6;
//  double B0Mass   = 5.27962;
  double B0Mass   = 5.27958;
  double B0Sigma  = 0.030;
  double JPsiMass = 3.096916;
  double PsiPMass = 3.686109;
  double piMass = 0.13957039;
  double kMass = 0.493677;
  double BpMass = 5.2791;
//  double KstarMass = 0.892;
  double KstarMass = 0.896;

  double XMinSBL  = 0.;
  double XMaxSBL  = 0.;
  double XMinSBR  = 0.;
  double XMaxSBR  = 0.;
//
  double NSigma1L = 0.;
  double NSigma2L = 0.;
  double NSigma1R = 0.;
  double NSigma2R = 0.;
  
  float HistMassL1 = 4.935;
  float HistMassL2 = 5.65;
//
//   double NSigmaSBL = -2.;
//   double NSigmaSBR = 0;
//   double BiasSB   = 6;
  double XMinCosThetaL	       = -1.;
  double XMaxCosThetaL	       =  1.;
  double XMinCosThetaK	       = -1.;
  double XMaxCosThetaK	       =  1.;
  double XMinPhi	       =-TMath::Pi();
  double XMaxPhi	       = TMath::Pi();

  double XMinCosThetaLUnfolded = -1.;
  double XMaxCosThetaLUnfolded =  1.;
  double XMinCosThetaKUnfolded = -1.;
  double XMaxCosThetaKUnfolded =  1.;
  double XMinPhiUnfolded       =-TMath::Pi();
  double XMaxPhiUnfolded       = TMath::Pi();

  int	 xCosLHBin =  25;
  int	 xCosKHBin =  25;
  int	 xPhiHBin  =  25;
  double BinWCosThetaL= 0;
  double BinWCosThetaK= 0;
  double BinWPhi      = 0;
  
  
  double xMinQMuMu = 1.;
  double xMaxQMuMu = 19.;
  double NSigma  = 3.;
//  double XMinSignW = XMinSign;
//  double XMaxSignW = XMaxSign;
//  double XMinSignW = B0Mass - NSigma*B0Sigma;
//  double XMaxSignW = B0Mass + NSigma*B0Sigma;
  double NSignInt2Sigma =0.;
  double NBckgInt2Sigma =0.;
  double XLeftSet =0.;
  double XRightSet =0.;
  double XStepSign = 0.0025;
  double XStepMinuit = 0.00001;
  float xMassHBin = (XMaxSign -XMinSign)/XStepSign;
  float xQ2HBin   = (xMaxQMuMu -xMinQMuMu)/0.1;
  double XHScale = 10;
 
  
  double yieldSignal = 0;
  double yieldBckg   = 0;
//   double ParMin = -1000;
//   double ParMax =  1000;
//   double RndMin = -0.1;
//   double RndMax =  0.1;
//  double c_const       = 0.0299792458;

  

  float xMassHBin2   =  xMassHBin /5; // plot only!
  GooFit::Observable xMass("xMass",XMinSign,XMaxSign) 	     ;
  
  TH1D* HxMass         = new TH1D( "HxMass"     , "B^{0} Mass"		 ,	      xMassHBin2, XMinSign,  XMaxSign);
  TH1D* HxMassQ2       = new TH1D( "HxMassQ2"   , "B^{0} Mass"		 ,	      xMassHBin2, XMinSign,  XMaxSign);
  TH1D* HxMassQ2SB     = new TH1D( "HxMassQ2SB" , "B^{0} Mass"		 ,	      xMassHBin2, XMinSign,  XMaxSign);
  TH1D* pdfHxMass      = new TH1D( "pdfHxMass"  , "B^{0} Mass Fit"	 ,  XHScale * xMassHBin , xMass.getLowerLimit(), xMass.getUpperLimit());
  TH1D* sigHxMass      = new TH1D( "sigHxMass"  , "B^{0} Mass Fit"	 ,  XHScale * xMassHBin , xMass.getLowerLimit(), xMass.getUpperLimit());
  TH1D* bkgHxMass      = new TH1D( "bkgHxMass"  , "B^{0} Mass Fit"	 ,  XHScale * xMassHBin , xMass.getLowerLimit(), xMass.getUpperLimit());
  TH1D* pdfHxMassQ2    = new TH1D( "pdfHxMassQ2", "B^{0} Mass Fit Q2 Bin",  XHScale * xMassHBin , xMass.getLowerLimit(), xMass.getUpperLimit());
  TH1D* sigHxMassQ2    = new TH1D( "sigHxMassQ2", "B^{0} Mass Fit Q2 Bin",  XHScale * xMassHBin , xMass.getLowerLimit(), xMass.getUpperLimit());
  TH1D* bkgHxMassQ2    = new TH1D( "bkgHxMassQ2", "B^{0} Mass Fit Q2 Bin",  XHScale * xMassHBin , xMass.getLowerLimit(), xMass.getUpperLimit());
//
  TH2D* HxMassVsCosL   = new TH2D( "HxMassVsCosL","B^{0} Mass%CosL"    ,(int)xMassHBin2, XMinSign,  XMaxSign, NFact*xCosLHBin, XMinCosThetaL, XMaxCosThetaL);
  TH2D* HxMassVsCosK   = new TH2D( "HxMassVsCosK","B^{0} Mass%CosK"    ,(int)xMassHBin2, XMinSign,  XMaxSign, NFact*xCosKHBin, XMinCosThetaK, XMaxCosThetaK);
  TH2D* HxMassVsPhi    = new TH2D( "HxMassVsPhi", "B^{0} Mass%Phi"     ,(int)xMassHBin2, XMinSign,  XMaxSign, NFact*xPhiHBin , XMinPhi, XMaxPhi);


//TH1D* pdfHist        = new TH1D( "pdfHist", "B^{0} Mass Fit",     xMassHBin2, XMinSign,  XMaxSign);
//TH1D* sigHist        = new TH1D( "sigHist", "B^{0} Mass Fit",     xMassHBin2, XMinSign,  XMaxSign);
//TH1D* bkgHist        = new TH1D( "bkgHist", "B^{0} Mass Fit",     xMassHBin2, XMinSign,  XMaxSign);
// Goofit Reco Observables..
//   GooFit::Observable xCosThetaL ("xCosThetaL"	    ,XMinCosThetaL,XMaxCosThetaL)    ;
//   GooFit::Observable xCosThetaK ("xCosThetaK"	    ,XMinCosThetaK,XMaxCosThetaK)    ;
//   GooFit::Observable xPhiKstMuMu("xPhiKstMuMu"      ,XMinPhi,XMaxPhi)  ; 
//  GooFit::Observable xQ2MuMu("xQ2MuMu",xMinQMuMu    ,xMaxQMuMu)                      ;

RooRealVar *tagged_mass=0;
RooRealVar *mumuMass=0;
RooRealVar *mumuMassE=0;
RooAbsPdf  *bkg_mass_sb           =0;
RooAbsPdf  *bkg_exp               =0;
TTree* RecoB0TreeOut  =0;
TFile*OutFileInputHisto;
double EffiFunc3D(Double_t *var, Double_t *par);
double EffiFunc2D(Double_t *var, Double_t *par);
GooFit::Application *app_ptr;
std::map<std::string, std::string>  ReadNamelist(int argc, char** argv);
//
TRatioPlot* RatioDataModel3DX = 0;
TRatioPlot* RatioDataModel3DY = 0; 
TRatioPlot* RatioDataModel3DZ = 0; 
//      
Minuit1 * Minuit = 0;
TMatrixD * covMatrix=0;
RooBernsteinSideband * BernSideBand =0;
RooRealVar* ctL = new RooRealVar("ctL", "ctL",  XMinCosThetaK,XMaxCosThetaK);
RooRealVar* ctK = new RooRealVar("ctK", "ctK",  XMinCosThetaL,XMaxCosThetaL);
RooRealVar* phi = new RooRealVar("phi", "phi",  XMinPhi,XMaxPhi);
//
//==========================================
// Adaptive Binning...
//==========================================
TKDTreeBinning* RecoAdaptBinsC = 0;
TKDTreeID* TKDTreeIDC =0;
int   xAdaptNumBinC = 1;
int   MinContAdaptBin = 5;  
int   NDim = 3;
//
//==========================================
//==========================================
//=========    MAIN    =====================
//==========================================
//==========================================

int main (int argc, char** argv) {
//gSystem->Load("libRIO.so");
//gSystem->Load("libTree.so");
gSystem->Load("libRooDoubleCBFast.so");
//gSystem->Load("RooBernsteinSideband.so");


// if (argc>1 ){
//     Q2Bin = (int) (*argv)[1];
// }
//     cout<<Q2Bin<<endl;
//     cout<<argv[0]<<endl;
//     exit(1);

if (argc<=1 ){
    cout<<"Q2Bin not set"<<endl;
    cout<<"Usage: "<<argv[0]<< " QBin2 [where QBin2=0,1,2,3,4,5,6,7,8]\n"<<endl;
    exit(1);
}   


 
switch ( *argv[1] ) {

  case '0' : 
   Q2Min = 1.; 
   Q2Max = 2.;
   Q2Bin = 0;
   if(RunEra==2016) CutSignificance =2;
   if(RunEra==2017) CutSignificance =2;
   if(RunEra==2018) CutSignificance =2;
//    xCosLHBin =   8;
//    xCosKHBin =   15;
//    xPhiHBin  =   8;
//       xCosLHBin =  25;
//       xCosKHBin =  25;
//       xPhiHBin  =  25;
// xCosLHBin =   5;
// xCosKHBin =   5;
// xPhiHBin  =   5;
    break;
  case '1' : 
   Q2Min = 2.; 
   Q2Max = 4.3; 
   Q2Bin = 1;
   if(RunEra==2016) CutSignificance =2;
   if(RunEra==2017) CutSignificance =2;
   if(RunEra==2018) CutSignificance =2;
//    xCosLHBin =  25;
//    xCosKHBin =  25;
//    xPhiHBin  =  25;
    break;
  case '2' : 
   Q2Min = 4.3; 
   Q2Max = 6.; 
   Q2Bin = 2;
   if(RunEra==2016) CutSignificance =2;
   if(RunEra==2017) CutSignificance =3;
   if(RunEra==2018) CutSignificance =3;
//    xCosLHBin =  5;
//    xCosKHBin =  5;
//    xPhiHBin  =  8;
//    xCosLHBin = 25;
//    xCosKHBin = 25;
//    xPhiHBin  = 25;
    break;
  case '3' : 
   Q2Min = 6.;  
   Q2Max = 8.68; 
   Q2Bin = 3;
   if(RunEra==2016) CutSignificance =2;
   if(RunEra==2017) CutSignificance =2;
   if(RunEra==2018) CutSignificance =3;
//    xCosLHBin = 25;
//    xCosKHBin = 25;
//    xPhiHBin  = 25;
//    xCosLHBin =  4;
//    xCosKHBin =  4;
//    xPhiHBin  =  7;
    break;
  case '4' : 
   Q2Min = 8.68; 
   Q2Max = 10.09; 
   Q2Bin = 4;
   sprintf(fitMassFileName,fitMassFileNameJpsi);
   if(RunEra==2016) CutSignificance =3;
   if(RunEra==2017) CutSignificance =3;
   if(RunEra==2018) CutSignificance =4.;
//   CutSignificance =4;
//    xCosLHBin =  5;
//    xCosKHBin =  5;
//    xPhiHBin  =  8;
//    xCosLHBin =  25;
//    xCosKHBin =  25;
//    xPhiHBin  =  25;
    break;
  case '5' :  
   Q2Min = 10.09; 
   Q2Max = 12.86; 
   Q2Bin = 5;
   if(RunEra==2016) CutSignificance =3;
   if(RunEra==2017) CutSignificance =3;
   if(RunEra==2018) CutSignificance =3;
//    xCosLHBin = 25;
//    xCosKHBin = 25;
//    xPhiHBin  = 25;
//    xCosLHBin =  5;
//    xCosKHBin =  5;
//    xPhiHBin  =  7;
    break;
  case '6' : 
   Q2Min = 12.86; 
   Q2Max = 14.18; 
   Q2Bin = 6;
   sprintf(fitMassFileName,fitMassFileNamePsi);
//   CutSignificance =4;
//    xCosLHBin = 25;
//    xCosKHBin = 25;
//    xPhiHBin  = 25;
    break;
  case '7' : 
   Q2Min = 14.18; 
   Q2Max = 16.; 
   Q2Bin = 7;
   sprintf(fitMassFileName,fitMassFileNameQ2Bin7);
   if(RunEra==2016) CutSignificance =2;
   if(RunEra==2017) CutSignificance =2;
   if(RunEra==2018) CutSignificance =2;
//    xCosLHBin = 25;
//    xCosKHBin = 25;
//    xPhiHBin  = 25;
    break;
  case '8' : 
   Q2Min = 16; 
   Q2Max = 19.; 
   Q2Bin = 8;
//    xCosLHBin = 25;
//    xCosKHBin = 25;
//    xPhiHBin  = 25;
    break;

  default : 
    // Process for all other cases.
    cout<<"Q2Bin not set correctly!!!"<<endl;
    cout<<"Usage: "<<argv[0]<< " QBin2 [where QBin2=0,1,2,3,4,5,6,7,8]\n"<<endl;
    exit(1);

}
//
//  Setting Input Files/Dir for RunEra..
//
replaceChar(RecoDir,"[RunEra]",Form("%d",RunEra));
replaceChar(InputFileNameRecoB0,"[RunEra]",Form("%d",RunEra));
replaceChar(fitMassFileName,"[RunEra]",Form("%d",RunEra));
//sprintf(fitMassFileName,"results_fits_%d_fixDataPdf.root",RunEra);
//======================================================================
//======================================================================
//===========		InputFileNameMCGene              ===============
//======================================================================
//======================================================================
//  sprintf(InputFileNameMCGene,"testGene-[RunEra]-Q2Bin-%d.root",Q2Bin);
//======================================================================
//======================================================================
//======================================================================
    std::cout<<"====================================="<<endl;
    
    

//   if (argc>3 && (strcmp(argv[3],"i") == 0) ){
//     integral = true;
//     sprintf(SigmaMethodTXT,"-integraBin");
//     std::cout<<"===================================================================="<<endl;
//     std::cout<<"======== DEFAULT: INTEGRAL of the SB function  =============="<<std::endl;
//     std::cout<<"===================================================================="<<endl;
//   }else{
//     integral = false;
//     sprintf(SigmaMethodTXT,"-centerBin");
//     std::cout<<"========================================================================="<<endl;
//     std::cout<<"Setting the option: SB function evaluated in the CENTER of the bin"<<std::endl;
//     std::cout<<"========================================================================="<<endl;
//   }
  char NameList[300];;
  
  
  
  sprintf(NameList,"namelist-SB3DB0-%d-Q2Bin-%d.lis",RunEra, Q2Bin);
//  
//   if(Folded){
//     sprintf(FoldedTXT,"-PhiFolded");
//     XMinPhi = 0.;
//     ParMin = 0.;
//     RndMin = 0.;
//     std::cout<<"===================================================================="<<endl;
//     std::cout<<"======== SETTING: Phi Ang.Variable FOLDED             =============="<<std::endl;
//     std::cout<<"===================================================================="<<endl;
//   };
  
  
  char*argn[]={NameList};
  
  std::map<std::string, std::string> mappa = ReadNamelist(1,argn );
//
  maxDegree1	    =	 atoi (mappa["maxDegree1"].c_str() ) ;
  maxDegree2	    =	 atoi (mappa["maxDegree2"].c_str() ) ;
  maxDegree3	    =	 atoi (mappa["maxDegree3"].c_str() ) ;
  xCosLHBin	    =	 atof (mappa["xCosLHBin" ].c_str() ) ;
  xCosKHBin	    =	 atof (mappa["xCosKHBin" ].c_str() ) ;
  xPhiHBin	    =	 atof (mappa["xPhiHBin"  ].c_str() ) ;
  NSigma1L	    =	 atof (mappa["NSigma1L"  ].c_str() ) ;
  NSigma2L	    =	 atof (mappa["NSigma2L"  ].c_str() ) ;
  NSigma1R	    =	 atof (mappa["NSigma1R"  ].c_str() ) ;
  NSigma2R	    =	 atof (mappa["NSigma2R"  ].c_str() ) ;
  maxDegree	    =	 atoi (mappa["maxDegree" ].c_str() ) ;
  fixParam	    =	 atoi (mappa["fixParam"  ].c_str() ) ; 
  map<string,string>::iterator  it= mappa.find("MinContAdaptBin");
  if(it != mappa.end()) {
   MinContAdaptBin   =    atoi (mappa["MinContAdaptBin"  ].c_str() ) ;
  }
  map<string,string>::iterator  it2= mappa.find("AutoFixPar");
  if(it2 != mappa.end()) {
   AutoFixPar   =    atol (mappa["AutoFixPar"  ].c_str() ) ;
  }
  if(AutoFixPar){
   std::cout<<"Warning: setting search for Param to fix = "<<AutoFixPar<<std::endl;
  }
  map<string,string>::iterator  it3= mappa.find("NFactGen");
  if(it3 != mappa.end()) {
   NFactGen   =    atoi (mappa["NFactGen"  ].c_str() ) ;
   std::cout<<"Warning: setting  NFactGen from namelist= "<<NFactGen<<std::endl;
  }
  map<string,string>::iterator  it4= mappa.find("SigmaProbSign");
  if(it4 != mappa.end()) {
   SigmaProbSign   =    atoi (mappa["SigmaProbSign"  ].c_str() ) ;
   std::cout<<"Warning: setting  SigmaProbSign from namelist= "<<SigmaProbSign<<std::endl;
  }
  map<string,string>::iterator  it7= mappa.find("tagged_mass_rangeMin");
  if(it7 != mappa.end()) {
   tagged_mass_rangeMin   =    atof (mappa["tagged_mass_rangeMin"  ].c_str() ) ;
   std::cout<<"Warning: setting  tagged_mass_rangeMin from namelist= "<<tagged_mass_rangeMin<<std::endl;
   if(tagged_mass_rangeMin<XMinSign){
    std::cout<<Form("Error: setting  tagged_mass_rangeMin=%f < XMinSign=%f",tagged_mass_rangeMin,XMinSign)<<std::endl;
    exit(0);
   }
  }
  map<string,string>::iterator  it8= mappa.find("tagged_mass_rangeMax");
  if(it8 != mappa.end()) {
   tagged_mass_rangeMax   =    atof (mappa["tagged_mass_rangeMax"  ].c_str() ) ;
   std::cout<<"Warning: setting  tagged_mass_rangeMax from namelist= "<<tagged_mass_rangeMax<<std::endl;
   if(tagged_mass_rangeMax>XMaxSign){
    std::cout<<Form("Error: setting  tagged_mass_rangeMax=%f < XMaxSign=%f",tagged_mass_rangeMax,XMaxSign)<<std::endl;
    exit(0);
   }
  }

  std::cout<<" Num Param Bernstein polynomial CosL :  "<<maxDegree1<<std::endl;
  std::cout<<" Num Param Bernstein polynomial CosK :  "<<maxDegree2<<std::endl;
  std::cout<<" Num Param Bernstein polynomial Phi  :  "<<maxDegree3<<std::endl;
  std::cout<<" Binning choice for CosL		   :  "<<xCosLHBin<<std::endl;
  std::cout<<" Binning choice for CosK		   :  "<<xCosKHBin<<std::endl;
  std::cout<<" Binning choice for Phi		   :  "<<xPhiHBin <<std::endl;
//
  std::cout<<" Min CosL XMinCosThetaL		   :  "<<XMinCosThetaL<<std::endl;
  std::cout<<" Max CosL XMaxCosThetaL		   :  "<<XMaxCosThetaL<<std::endl;
  std::cout<<" Min CosK XMinCosThetaK		   :  "<<XMinCosThetaK<<std::endl;
  std::cout<<" Min CosK XMaxCosThetaK		   :  "<<XMaxCosThetaK<<std::endl;
  std::cout<<" Min Phi  XMinPhi 		   :  "<<XMinPhi<<std::endl;
  std::cout<<" Min Phi  XMaxPhi 		   :  "<<XMaxPhi<<std::endl;
//
  if(SigmaProbSign==0){
   sprintf(SigmaMethodTXT,"-SigmaGauss");
   std::cout<<" NSigma1L [sigma gauss model]:  "<<NSigma1L<<std::endl;
   std::cout<<" NSigma2L [sigma gauss model]:  "<<NSigma2L<<std::endl;
   std::cout<<" NSigma1R [sigma gauss model]:  "<<NSigma1R<<std::endl;
   std::cout<<" NSigma2R [sigma gauss model]:  "<<NSigma2R<<std::endl;
  }else if(SigmaProbSign==-1){
   sprintf(SigmaMethodTXT,"-SigmaBare");
   std::cout<<" NSigma1L [bare min limit left ]:"<<NSigma1L<<std::endl;
   std::cout<<" NSigma2L [bare max limit left ]:"<<NSigma2L<<std::endl;
   std::cout<<" NSigma1R [bare min limit right]:"<<NSigma1R<<std::endl;
   std::cout<<" NSigma2R [bare max limit right]:"<<NSigma2R<<std::endl;
  }else if (SigmaProbSign==1){
   sprintf(SigmaMethodTXT,"-SigmaProb");
   std::cout<<" NSigma1L [Limit in GeV Left ]:  "<<NSigma1L<<std::endl;
   std::cout<<" NSigma2L [n. gauss stand. dev. sign]:  "<<NSigma2L<<std::endl;
   std::cout<<" NSigma1R [n. gauss stand. dev. sign]:  "<<NSigma1R<<std::endl;
   std::cout<<" NSigma2R [Limit in GeV Right]:  "<<NSigma2R<<std::endl;
  }else{
   std::cout<<Form(" SigmaProbSign: 	INVALID OPTION: %f !!! Exit...",SigmaProbSign)<<std::endl;
   exit(1);
  }
//
  std::cout<<" Num Param Bernstein polynomial Mass :  "<<maxDegree<<std::endl;
  std::cout<<" Parameter to Fix for normalization  :  "<<fixParam<<std::endl;
  std::cout<<" Num of events inside adaptive bins  :  "<<MinContAdaptBin<<std::endl;
  
//
  if ((SigmaProbSign==-1)&&
       (NSigma1L==0.0 ||
        NSigma2L==0.0 ||
        NSigma1R==0.0 ||
        NSigma2R==0.0)){
     std::cout<<"====================================================================================="<<std::endl;
     std::cout<<"====================================================================================="<<std::endl;
     std::cout<<"====================================================================================="<<std::endl;
     std::cout<<"Error reading sideband limits: at least one NSigma[]==0 found !!!!"<<std::endl;
     std::cout<<"====> EXIT from Main!!!"<<std::endl;
     std::cout<<"====================================================================================="<<std::endl;
     std::cout<<"====================================================================================="<<std::endl;
     std::cout<<"====================================================================================="<<std::endl;
     exit(0);
  }    


//



  char TXTNSigma1L[10]="";
  char TXTNSigma2L[10]="";
  char TXTNSigma1R[10]="";
  char TXTNSigma2R[10]="";
  
  std::stringstream sss;
  std::string ss;
  std::string dot1=".";
  std::string dot2="dot";

  sss<<NSigma1L;
  ss=sss.str();
  cout<<ss<<endl;
  strcpy(FMTNSigma1L,ss.c_str());
  replaceAll( ss,  dot1, dot2);
  strcpy(TXTNSigma1L,ss.c_str());
  sss.str("");
  sss.clear();
  sss<<NSigma1R;
  ss=sss.str();
  cout<<ss<<endl;
  strcpy(FMTNSigma1R,ss.c_str());
  replaceAll( ss,  dot1, dot2);
  strcpy(TXTNSigma1R,ss.c_str());
  sss.str("");
  sss.clear();
  sss<<NSigma2L;
  ss=sss.str();
  cout<<ss<<endl;
  strcpy(FMTNSigma2L,ss.c_str());
  replaceAll( ss,  dot1, dot2);
  strcpy(TXTNSigma2L,ss.c_str());
  sss.str("");
  sss.clear();
  sss<<NSigma2R;
  ss=sss.str();
  cout<<ss<<endl;
  strcpy(FMTNSigma2R,ss.c_str());
  replaceAll( ss,  dot1, dot2);
  strcpy(TXTNSigma2R,ss.c_str());
   
//   cout<<TXTNSigma1L<<endl;
//   cout<<FMTNSigma1L<<endl;
//   cout<<TXTNSigma1R<<endl;
//   cout<<FMTNSigma1R<<endl;
//   cout<<TXTNSigma2L<<endl;
//   cout<<FMTNSigma2L<<endl;
//   cout<<TXTNSigma2R<<endl;
//   cout<<FMTNSigma2R<<endl;

  std::cout<<"--------------------------------------------\n"<<endl;
  std::cout<<" Setting selection for q^2 bin: "<<*argv[1]<<" ==> "<<Q2Min<<"<q^2<"<<Q2Max<<std::endl;
  std::cout<<"--------------------------------------------\n"<<endl;
  sprintf(OutFileName,"testGoofitSB3DB0-%d%s-Q2Bin-%d-Bins-%d-%d-%d-BernDeg-%d-%d-%d-WSBL-%s-%s-WSBR-%s-%s%s%s%s.root",RunEra,ProjectTXT, Q2Bin,xCosLHBin,xCosKHBin,xPhiHBin,maxDegree1,maxDegree2,maxDegree3,TXTNSigma1L,TXTNSigma2L,TXTNSigma1R,TXTNSigma2R,TaggedVarTXT,SigmaMethodTXT,FoldedTXT); 
  sprintf(ListParName,"ListParValues-%d-%s-Q2Bin-%d-Bins-%d-%d-%d-BernDeg-%d-%d-%d-WSBL-%s-%s-WSBR-%s-%s%s%s%s.txt"   ,RunEra,ProjectTXT, Q2Bin,xCosLHBin,xCosKHBin,xPhiHBin,maxDegree1,maxDegree2,maxDegree3,TXTNSigma1L,TXTNSigma2L,TXTNSigma1R,TXTNSigma2R,TaggedVarTXT,SigmaMethodTXT,FoldedTXT); 
  sprintf(ListPloName,"ListParValues-%d-%s-Q2Bin-%d-Bins-%d-%d-%d-BernDeg-%d-%d-%d-WSBL-%s-%s-WSBR-%s-%s%s%s%s.plo"   ,RunEra,ProjectTXT, Q2Bin,xCosLHBin,xCosKHBin,xPhiHBin,maxDegree1,maxDegree2,maxDegree3,TXTNSigma1L,TXTNSigma2L,TXTNSigma1R,TXTNSigma2R,TaggedVarTXT,SigmaMethodTXT,FoldedTXT); 
  sprintf(PDFNameFitSB3D,"B0-FitSB3D-%d-%s-Q2Bin-%d-Bins-%d-%d-%d-BernDeg-%d-%d-%d-WSBL-%s-%s-WSBR-%s-%s%s%s%s-Adapt-%d.pdf",RunEra,ProjectTXT,  Q2Bin,xCosLHBin,xCosKHBin,xPhiHBin,maxDegree1,maxDegree2,maxDegree3,TXTNSigma1L,TXTNSigma2L,TXTNSigma1R,TXTNSigma2R,TaggedVarTXT,SigmaMethodTXT,FoldedTXT,MinContAdaptBin); 
  sprintf(PNGNameFitSB3D,"B0-FitSB3D-%d-%s-Q2Bin-%d-Bins-%d-%d-%d-BernDeg-%d-%d-%d-WSBL-%s-%s-WSBR-%s-%s%s%s%s-Adapt-%d.png",RunEra,ProjectTXT,  Q2Bin,xCosLHBin,xCosKHBin,xPhiHBin,maxDegree1,maxDegree2,maxDegree3,TXTNSigma1L,TXTNSigma2L,TXTNSigma1R,TXTNSigma2R,TaggedVarTXT,SigmaMethodTXT,FoldedTXT,MinContAdaptBin); 
  sprintf(PNGNameMassQ2Hist,"B0-MassQ2-%d-%s-Q2Bin-%d-Bins-%d-%d-%d-BernDeg-%d-%d-%d-WSBL-%s-%s-WSBR-%s-%s%s%s%s.png" ,RunEra,ProjectTXT, Q2Bin,xCosLHBin,xCosKHBin,xPhiHBin,maxDegree1,maxDegree2,maxDegree3,TXTNSigma1L,TXTNSigma2L,TXTNSigma1R,TXTNSigma2R,TaggedVarTXT,SigmaMethodTXT,FoldedTXT); 
  sprintf(PNGNameMassHist,"B0-MassTot-%d-%s-Q2Bin-%d-Bins-%d-%d-%d-BernDeg-%d-%d-%d-WSBL-%s-%s-WSBR-%s-%s%s%s%s.png"  ,RunEra,ProjectTXT, Q2Bin,xCosLHBin,xCosKHBin,xPhiHBin,maxDegree1,maxDegree2,maxDegree3,TXTNSigma1L,TXTNSigma2L,TXTNSigma1R,TXTNSigma2R,TaggedVarTXT,SigmaMethodTXT,FoldedTXT); 
  sprintf(PNGNameMassCheck,"B0-MassCheck-%d-%s-Q2Bin-%d-Bins-%d-%d-%d-BernDeg-%d-%d-%d-WSBL-%s-%s-WSBR-%s-%s%s%s%s.png"  ,RunEra,ProjectTXT, Q2Bin,xCosLHBin,xCosKHBin,xPhiHBin,maxDegree1,maxDegree2,maxDegree3,TXTNSigma1L,TXTNSigma2L,TXTNSigma1R,TXTNSigma2R,TaggedVarTXT,SigmaMethodTXT,FoldedTXT); 
  sprintf(PNGNameFitSB3DMass,"B0-FitSB3D-Mass-%d-%s-Q2Bin-%d-Bins-%d-%d-%d-BernDeg-%d-%d-%d-WSBL-%s-%s-WSBR-%s-%s%s%s%s-Adapt-%d.png",RunEra,ProjectTXT,  Q2Bin,xCosLHBin,xCosKHBin,xPhiHBin,maxDegree1,maxDegree2,maxDegree3,TXTNSigma1L,TXTNSigma2L,TXTNSigma1R,TXTNSigma2R,TaggedVarTXT,SigmaMethodTXT,FoldedTXT,MinContAdaptBin); 
  sprintf(PNGNameFitSB3DProjX,"B0-FitSB3D-ProjX-%d-%s-Q2Bin-%d-Bins-%d-%d-%d-BernDeg-%d-%d-%d-WSBL-%s-%s-WSBR-%s-%s%s%s%s-Adapt-%d.png",RunEra,ProjectTXT,  Q2Bin,xCosLHBin,xCosKHBin,xPhiHBin,maxDegree1,maxDegree2,maxDegree3,TXTNSigma1L,TXTNSigma2L,TXTNSigma1R,TXTNSigma2R,TaggedVarTXT,SigmaMethodTXT,FoldedTXT,MinContAdaptBin); 
  sprintf(PNGNameFitSB3DProjY,"B0-FitSB3D-ProjY-%d-%s-Q2Bin-%d-Bins-%d-%d-%d-BernDeg-%d-%d-%d-WSBL-%s-%s-WSBR-%s-%s%s%s%s-Adapt-%d.png",RunEra,ProjectTXT,  Q2Bin,xCosLHBin,xCosKHBin,xPhiHBin,maxDegree1,maxDegree2,maxDegree3,TXTNSigma1L,TXTNSigma2L,TXTNSigma1R,TXTNSigma2R,TaggedVarTXT,SigmaMethodTXT,FoldedTXT,MinContAdaptBin); 
  sprintf(PNGNameFitSB3DProjZ,"B0-FitSB3D-ProjZ-%d-%s-Q2Bin-%d-Bins-%d-%d-%d-BernDeg-%d-%d-%d-WSBL-%s-%s-WSBR-%s-%s%s%s%s-Adapt-%d.png",RunEra,ProjectTXT,  Q2Bin,xCosLHBin,xCosKHBin,xPhiHBin,maxDegree1,maxDegree2,maxDegree3,TXTNSigma1L,TXTNSigma2L,TXTNSigma1R,TXTNSigma2R,TaggedVarTXT,SigmaMethodTXT,FoldedTXT,MinContAdaptBin); 
//
  sprintf(PNGNameProjXYHist,"B0-SB-ProjXY-%d-%s-Q2Bin-%d-Bins-%d-%d-%d-BernDeg-%d-%d-%d-WSBL-%s-%s-WSBR-%s-%s%s%s%s.png"  ,RunEra,ProjectTXT, Q2Bin,xCosLHBin,xCosKHBin,xPhiHBin,maxDegree1,maxDegree2,maxDegree3,TXTNSigma1L,TXTNSigma2L,TXTNSigma1R,TXTNSigma2R,TaggedVarTXT,SigmaMethodTXT,FoldedTXT); 
  sprintf(PNGNameProjZYHist,"B0-SB-ProjZY-%d-%s-Q2Bin-%d-Bins-%d-%d-%d-BernDeg-%d-%d-%d-WSBL-%s-%s-WSBR-%s-%s%s%s%s.png"  ,RunEra,ProjectTXT, Q2Bin,xCosLHBin,xCosKHBin,xPhiHBin,maxDegree1,maxDegree2,maxDegree3,TXTNSigma1L,TXTNSigma2L,TXTNSigma1R,TXTNSigma2R,TaggedVarTXT,SigmaMethodTXT,FoldedTXT); 
  sprintf(PNGNameProjZXHist,"B0-SB-ProjZX-%d-%s-Q2Bin-%d-Bins-%d-%d-%d-BernDeg-%d-%d-%d-WSBL-%s-%s-WSBR-%s-%s%s%s%s.png"  ,RunEra,ProjectTXT, Q2Bin,xCosLHBin,xCosKHBin,xPhiHBin,maxDegree1,maxDegree2,maxDegree3,TXTNSigma1L,TXTNSigma2L,TXTNSigma1R,TXTNSigma2R,TaggedVarTXT,SigmaMethodTXT,FoldedTXT); 
//  sprintf(fitMassFileName,"save-w-%d-bin-%d.root",Q2Bin);
  sprintf(OutSaveFileName,"savesb-%d%s-Q2Bin-%d-Bins-%d-%d-%d-BernDeg-%d-%d-%d-WSBL-%s-%s-WSBR-%s-%s%s%s%s.root",RunEra,ProjectTXT, Q2Bin,xCosLHBin,xCosKHBin,xPhiHBin,maxDegree1,maxDegree2,maxDegree3,TXTNSigma1L,TXTNSigma2L,TXTNSigma1R,TXTNSigma2R,TaggedVarTXT,SigmaMethodTXT,FoldedTXT); 

  sprintf(OutFileNameInputHisto,"testGoofitSB3DB0-%d-InputHisto-Q2Bin-%d-Bins-%d-%d-%d-masspectrum%s.root",RunEra,Q2Bin,xCosLHBin,xCosKHBin,xPhiHBin,FoldedTXT); 



  
  BinWCosThetaL=(XMaxCosThetaL-XMinCosThetaL)/double(xCosLHBin);
  BinWCosThetaK=(XMaxCosThetaK-XMinCosThetaK)/double(xCosKHBin);
  BinWPhi      =(XMaxPhi-XMinPhi)/double(xPhiHBin);

//  TApplication tapp("TApp",&argc, argv);
  GooFit::Application app("testGoofit3DB0-[RunEra] fit example", argc, argv);
  app_ptr = &app;
//  app.require_subcommand();

//  app.add_flag("--minuit1", minuit1, "Use Minuit 1 instead of Minuit 2");
 
  TStopwatch TimeWatch;
  TimeWatch.Start();

  FitSBModel(); 
//  app.Run() ;
  cout<<"esco..." <<endl;
  TimeWatch.Stop();
  TimeWatch.Print();
  
//  GOOFIT_PARSE(app);
  return 0 ;
}


void FitSBModel(){


  gettimeofday(&startTime, NULL);
  startCPU = times(&startProc);

   gROOT->Reset();
   gROOT->SetStyle("Plain");
   gROOT->ForceStyle();
   gStyle->SetOptStat(000000);
   gStyle->SetOptFit(000000);
   gStyle->SetPadBorderMode(0);

   TCanvas* c1 = new TCanvas("c1","Mass",200,10,900,780);
   TCanvas* c2 = new TCanvas("c2","Fit Mass Spectrum",200,10,900,780);
   TCanvas* cc = new TCanvas("cc","Fit Mass Spectrum Check",200,10,900,780);
//    TCanvas* c5 = new TCanvas("c5","                      ",200,10,900,780);
   TCanvas* c6 = new TCanvas("c6","Sideband",200,10,900,900);
   TCanvas* cmass = new TCanvas("cmas","Mass",200,10,750,800);
   TCanvas* cprojX = new TCanvas("cprojX","Angular Projections",200,10,900,780);
   TCanvas* cprojY = new TCanvas("cprojY","Angular Projections",200,10,900,780);
   TCanvas* cprojZ = new TCanvas("cprojZ","Angular Projections",200,10,900,780);
   TCanvas* cxy = new TCanvas("cxy","Sideband",200,10,750,750);
   TCanvas* cyz = new TCanvas("cyz","Sideband",200,10,750,780);
   TCanvas* cxz = new TCanvas("czy","Sideband",200,10,750,780);
   csignstudy = new TCanvas("csignstudy","Mass Signal study",200,10,900,450);
   c6->Divide(2,2);  
   csignstudy->Divide(2,1);
//    TCanvas* c7 = new TCanvas("c7","Efficiencies" ,200,10,900,780);
//    TCanvas* c8 = new TCanvas("c8","Effi-Reco Closure Test" ,200,10,900,780);
//   c2->Divide(2,2);  
//    c5->Divide(2,2);  
//    c7->Divide(2,2);  
//    c8->Divide(2,2);  
//    TPad* pad2 = (TPad*)c2->GetPad(0);
//    pad2->SetLeftMargin(0.15); 
//    pad2->SetRightMargin(0.15); 
//    TPad* pad6 = (TPad*)c6->GetPad(0);
//    pad6->SetLeftMargin(0.15); 
//    pad6->SetRightMargin(0.15); 

   //TPad* pad1 = (TPad*)c1->GetPad(0);
//   TPad* pad2 = (TPad*)c2->GetPad(0);
   //pad1->SetLeftMargin(0.15); 
//   pad2->SetLeftMargin(0.15); 



//  gSystem->Exec(Form("mv %s %s.tmp",OutFileName,OutFileName));
//  OutFileInputHisto = TFile::Open(OutFileNameInputHisto,"READ");
  if (!TFile::Open(OutFileNameInputHisto,"READ"))
  {
    cout<<"File:"<<OutFileNameInputHisto<<" not found!!! create..."<<endl;
    CreateInputHistoFile();
    OutFileInputHisto = TFile::Open(OutFileNameInputHisto,"READ");
  }else{
   OutFileInputHisto = TFile::Open(OutFileNameInputHisto,"READ");
   cout<<"File:"<<OutFileNameInputHisto<<" FOUND !!!"<<endl;
  } 
  gSystem->Exec(Form("mv %s %s.tmp",OutFileName,OutFileName));
  OutFile = TFile::Open(OutFileName,"RECREATE");
  TTree *RecoB0TreeOut     = (TTree*)OutFileInputHisto->Get(OutputRecoB0TreeName);
   if(!RecoB0TreeOut ){
     cout<<"TTree Reco Data: "<< OutputRecoB0TreeName <<" not found!!! Suggestion: remove this file e try again..."<<endl;
     exit(1);
   }else{
     cout<<"TTree Reco Data: "<< OutputRecoB0TreeName <<" OK FOUND!!!"<<endl;
   }  
  HxMass    = (TH1D*)OutFileInputHisto->Get("HxMass");
  if(!HxMass ){
    cout<<"HxMass Histo: not found!!! Exit..."<<endl;
    exit(1);
  }else{
    cout<<"HxMass Histo: OK FOUND!!! Entries: "<<HxMass->GetEntries()<<endl;
  } 
//    HxMassQ2    = (TH1D*)OutFileInputHisto->Get("HxMassQ2");
//   if(!HxMassQ2 ){
//     cout<<"HxMassQ2 Histo: not found!!! Exit..."<<endl;
//     exit(1);
//   }else{
//     cout<<"HxMassQ2 Histo: OK FOUND!!! Entries: "<<HxMassQ2->GetEntries()<<endl;
//   } 
 

  
//   HxReco    = (TH3D*)OutFileInputHisto->Get("HxReco");
//   if(!HxReco ){
//     cout<<"HxReco Histo: not found!!! Exit..."<<endl;
//     exit(1);
//   }else{
//     cout<<"HxReco Histo: OK FOUND!!! Entries: "<<HxReco->GetEntries()<<endl;
//     if(HxReco->GetNbinsX()!=xCosLHBin){cout<<"Error HxReco NBinsX = "<<HxReco->GetNbinsX()<<" != xCosLHBin = "<<xCosLHBin<<endl;exit(1);}
//     if(HxReco->GetNbinsY()!=xCosKHBin){cout<<"Error HxReco NBinsY = "<<HxReco->GetNbinsY()<<" != xCosKHBin = "<<xCosKHBin<<endl;exit(1);}
//     if(HxReco->GetNbinsZ()!=xPhiHBin ){cout<<"Error HxReco NBinsZ = "<<HxReco->GetNbinsZ()<<" != xPhiHBin  = "<<xPhiHBin <<endl;exit(1);}
//   }  
  cout<<"======================="<<endl;
  cout<<"xCosLHBin = "<<xCosLHBin<<endl;
  cout<<"xCosKHBin = "<<xCosKHBin<<endl;
  cout<<"xPhiHBin  = "<<xPhiHBin <<endl; 
  cout<<"======================="<<endl;
//
  TH3D* HxReco = new   TH3D( "HxReco"    , "B^{0} Reco correct tagged",  xCosLHBin, XMinCosThetaL, XMaxCosThetaL,
 									 xCosKHBin, XMinCosThetaK, XMaxCosThetaK,
									 xPhiHBin , XMinPhi, XMaxPhi );


  std::vector<GooFit::Observable> dataVec;
  GooFit::Observable xCosL_x("xCosL_x"  ,XMinCosThetaL, XMaxCosThetaL)  ;
  GooFit::Observable xCosK_y("xCosK_y"  ,XMinCosThetaK, XMaxCosThetaK)  ;
  GooFit::Observable xPhiK_z("xPhiK_z"  ,XMinPhi, XMaxPhi)  ;
//   GooFit::Observable xBinWidth1("xBinWidth1", 0.,100.);
//   GooFit::Observable yBinWidth1("yBinWidth1", 0.,100.);
//   GooFit::Observable zBinWidth1("zBinWidth1", 0.,100.);
  dataVec.push_back(xCosL_x);
  dataVec.push_back(xCosK_y);
  dataVec.push_back(xPhiK_z);
  UnbinnedDataSet* dataReco = new GooFit::UnbinnedDataSet(dataVec);
  std::vector<GooFit::Observable> plotVec;
  double cos_theta_l	;
  double cos_theta_k	;  
  double phi_kst_mumu	;
  double xtagged_mass	;
  double xmumuMass	;
  double xmumuMassE     ;
//  double mmk1	        ;
//  double mmk2           ;
  RecoB0TreeOut->SetBranchAddress("cos_theta_l"   ,&cos_theta_l );
  RecoB0TreeOut->SetBranchAddress("cos_theta_k"   ,&cos_theta_k );
  RecoB0TreeOut->SetBranchAddress("phi_kst_mumu"  ,&phi_kst_mumu);
  RecoB0TreeOut->SetBranchAddress("tagged_mass"   ,&xtagged_mass );
  RecoB0TreeOut->SetBranchAddress("mumuMass"      ,&xmumuMass );
  RecoB0TreeOut->SetBranchAddress("mumuMassE"     ,&xmumuMassE );
//  RecoB0TreeOut->SetBranchAddress("mmk1"          ,&mmk1 );
//  RecoB0TreeOut->SetBranchAddress("mmk2"          ,&mmk2 );
  int nentries = (int)RecoB0TreeOut->GetEntries();
//  int nentries = 0;
  cout<<"nentries: "<< nentries<<endl;
//  nentries = nentries/2.;
//  cout<<"PADUL!!!!! half nentries: "<< nentries<<endl;
  std::vector<GooFit::Observable> dataMassSBVec;
  dataMassSBVec.push_back(xMass);
  UnbinnedDataSet* dataMassSB = new GooFit::UnbinnedDataSet(dataMassSBVec);
  tagged_mass = new RooRealVar("tagged_mass" , "#mu^{+}#mu^{-}K#pi mass", XMinSign, XMaxSign, "GeV");
  mumuMass    = new RooRealVar("mumuMass"    , "mumuMass" , 0, 6);
  mumuMassE   = new RooRealVar("mumuMassE"   , "mumuMassE", 0, 10000);
  RooDataSet *fulldata   = new RooDataSet("fulldata", "fulldataset",  RooArgSet(*tagged_mass,*mumuMass,*mumuMassE));
  for (Int_t i=0;i<nentries;i++) {
         RecoB0TreeOut->GetEntry(i);
//  	 if(mmk2>3.6&&mmk2<4.2&&mmk1>4.7&&mmk1<4.9&&Q2Bin==4)continue;
         if( (xtagged_mass>XMinSign&&xtagged_mass<XMaxSign) ){
	   HxMassQ2->Fill(xtagged_mass);
 	   xMass.setValue(xtagged_mass);
 	   dataMassSB->addEvent();
	   tagged_mass->setVal(xtagged_mass);
	   mumuMass->setVal(xmumuMass);
	   mumuMassE->setVal(xmumuMassE);
	   fulldata->add(RooArgSet(*tagged_mass,*mumuMass,*mumuMassE));
	 } 
  }

  if(Q2Bin==4){  
    double B0Sigma_tmp = FitMassSpectrum(dataMassSB, cc, HxMassQ2,pdfHxMassQ2,sigHxMassQ2,bkgHxMassQ2, maxDegree);
    gSystem->Exec(Form("mv %s %s.tmp",PNGNameMassCheck,PNGNameMassCheck));
    cc->Print(PNGNameMassCheck);
  }  
//  std::cout<< "Setting B0Sigma = "<<B0Sigma<<" from the fit to the mass spectrum\n"<<std::endl; 
  B0Sigma = FitMassSpectrumRoofit(fulldata, c2, HxMassQ2,pdfHxMassQ2,sigHxMassQ2,bkgHxMassQ2, maxDegree);
//  exit(1);
  std::cout<< "Setting B0Sigma = "<<B0Sigma<<" from the fit to the mass spectrum\n"<<std::endl; 
//  c2->Write();
  gSystem->Exec(Form("mv %s %s.tmp",PNGNameMassQ2Hist,PNGNameMassQ2Hist));
  c2->Print(PNGNameMassQ2Hist);
//   XMinSBL = B0Mass - NSigma1L*B0Sigma;
//   XMaxSBL = B0Mass - NSigma2L*B0Sigma;
//   XMinSBR = B0Mass + NSigma1R*B0Sigma;
//   XMaxSBR = B0Mass + NSigma2R*B0Sigma;
  std::cout<<" XMinSBL  			 :     "<<XMinSBL<<std::endl;
  std::cout<<" XMaxSBL  			 :     "<<XMaxSBL<<std::endl;
  std::cout<<" XMinSBR  			 :     "<<XMinSBR<<std::endl;
  std::cout<<" XMaxSBR  			 :     "<<XMaxSBR<<std::endl;
  
  std::vector<double> CorreAdaptX;
  std::vector<double> CorreAdaptY;
  std::vector<double> CorreAdaptZ;
  for (Int_t i=0;i<nentries;i++) {
  	  RecoB0TreeOut->GetEntry(i);
	  if(cos_theta_l==-99) continue;
//    	  if(mmk2>3.6&&mmk2<4.2&&mmk1>4.7&&mmk1<4.9&&Q2Bin==4)continue;
// 	  if(cos_theta_l>XMaxCosThetaL) continue;
// 	  if(cos_theta_l<XMinCosThetaL) continue;
// 	  if(cos_theta_k>XMaxCosThetaK) continue;
// 	  if(cos_theta_k<XMinCosThetaK) continue;
// 	  if(phi_kst_mumu<XMinPhi     ) continue;
// 	  if(phi_kst_mumu>XMaxPhi     ) continue;
          if( (xtagged_mass>XMinSBL&&xtagged_mass<XMaxSBL)|| 
	      (xtagged_mass>XMinSBR&&xtagged_mass<XMaxSBR)){
	   HxMassQ2SB->Fill(xtagged_mass);
	   HxMassVsCosL->Fill(xtagged_mass,cos_theta_l);
	   HxMassVsCosK->Fill(xtagged_mass,cos_theta_k);
	   HxMassVsPhi ->Fill(xtagged_mass,phi_kst_mumu);
     	   xCosL_x.setValue(cos_theta_l);
     	   xCosK_y.setValue(cos_theta_k);
     	   xPhiK_z.setValue(phi_kst_mumu);
	   HxReco->Fill(cos_theta_l,cos_theta_k,phi_kst_mumu);
// 	   double xL = xCosL_x.getValue();
//            double yK = xCosK_y.getValue();
//            double zP = xPhiK_z.getValue();

           dataReco->addEvent();
//	   std::cout<<xL<<" "<<yK<<" "<<zP<<std::endl;
	   CorreAdaptX.push_back(cos_theta_l);
	   CorreAdaptY.push_back(cos_theta_k);
	   CorreAdaptZ.push_back(phi_kst_mumu);
         }
  }
  std::cout<<"Found SB entries = "<<HxReco->GetEntries()<<std::endl;
  if(HxReco->GetEntries()<10){
   std::cout<<"Error!! too few SB entries for a Fit: SB entries =  "<<HxReco->GetEntries()<<" EXIT!!!"<<std::endl;
   exit(0);
  }
  double xBinw =  HxReco->GetXaxis()->GetBinWidth(1) ;
  double yBinw =  HxReco->GetYaxis()->GetBinWidth(1) ;
  double zBinw =  HxReco->GetZaxis()->GetBinWidth(1) ;
  TH1D* HxRecoCosL = (TH1D*) HxReco->Project3D("x");
  TH1D* HxRecoCosK = (TH1D*) HxReco->Project3D("y");
  TH1D* HxRecoPhi  = (TH1D*) HxReco->Project3D("z");
// 
  TH2D* HxRecoCosLK =(TH2D*) HxReco->Project3D("xy");
  
//   plotVec.push_back(xCosL_x);
//   plotVec.push_back(xCosK_y);
//   plotVec.push_back(xPhiK_z);
//   UnbinnedDataSet* dataPlot = new GooFit::UnbinnedDataSet(plotVec);
//   
//   
  vector<GooFit::Observable> obsPoly;
  obsPoly.push_back(xCosL_x);
  obsPoly.push_back(xCosK_y);
  obsPoly.push_back(xPhiK_z);

//   vector<GooFit::Observable> obsPolyPlot;
//   obsPolyPlot.push_back(xCosL_x);
//   obsPolyPlot.push_back(xCosK_y);
//   obsPolyPlot.push_back(xPhiK_z);

  GooFit::Variable XMinCosL( "XMinCosL" , XMinCosThetaL);
  GooFit::Variable XMaxCosL( "XMaxCosL" , XMaxCosThetaL);
  GooFit::Variable XMinCosK( "XMinCosK" , XMinCosThetaK);
  GooFit::Variable XMaxCosK( "XMaxCosK" , XMaxCosThetaK);
  GooFit::Variable XMinPhiK( "XMinPhiK" , XMinPhi);
  GooFit::Variable XMaxPhiK( "XMaxPhiK" , XMaxPhi);
//  
  GooFit::Variable xBinWidth("xBinWidth", xBinw);
  GooFit::Variable yBinWidth("yBinWidth", yBinw);
  GooFit::Variable zBinWidth("zBinWidth", zBinw);
//  
  vector<GooFit::Variable> limits;
  limits.push_back(XMinCosL);
  limits.push_back(XMaxCosL);
  limits.push_back(XMinCosK);
  limits.push_back(XMaxCosK);
  limits.push_back(XMinPhiK);
  limits.push_back(XMaxPhiK);
  
  vector<GooFit::Variable> Binws;
  Binws.push_back(xBinWidth);
  Binws.push_back(yBinWidth);
  Binws.push_back(zBinWidth);



  int	numParameters = (maxDegree1+1)*(maxDegree2+1)*(maxDegree3+1);
  int icc=0;
  char ParCheck[numParameters][30];
  for(int i = 0; i <= maxDegree1 ; ++i) {
    for(int j = 0; j <= maxDegree2 ; ++j) {
     for(int k = 0; k <= maxDegree3 ; ++k) {
            
      	    sprintf(ParCheck[icc], "cosL=%d cosK=%d phi=%d", i,j,k); 
//	    cout<<Form("ParCheck(%d)= %s",icc,ParCheck[icc])<<endl;
	    icc++;
	    
     }
    }
  }
  vector<GooFit::Variable> coeffPoly;
//  int lim = limits.size();
  char varName[100];
  int NumCalls = 150000;
  
  int NumParamFree = 0;
  int fixParamTmp = -999;
  double parIni=0.;
  bool FoundParamNotZero = false;
  bool SearchFixParam = true;
  std::string line;
//  std::size_t sz;
//  double NParIni = 100.;
  std::cout<<"Try to open list of initial parameters :"<< ListParName <<std::endl;
  std::fstream *parListInput = new std::fstream(ListParName,std::ifstream::in);
  if(parListInput->is_open()){
     std::cout<<"List of initial parameters :"<< ListParName <<" FOUND!!!"<<std::endl;
     for (int i=0;i<numParameters;++i){
      	    sprintf(varName, "p%d", i);
            std::getline(*parListInput, line);
	    char* pEnd;
	    parIni =  strtod(line.c_str(), &pEnd);
//	    parIni =  stod(line, sz);
	    
// 	    *parListInput >> parIni;
//	    getline (parList,line);
     			if(fabs(parIni)>0.0009999) {
			 
			 if(parIni==1.00000) fixParamTmp = i;
//			 std::cout<<"parIni = "<<parIni<<std::endl;
//     			if(parIni!=0.0 && fabs(parIni)>0.00001) {
//     			 coeffPoly.emplace_back(varName, parIni,-100.0+parIni,100.0+parIni);
     			 coeffPoly.emplace_back(varName, parIni,ParMin,ParMax);
			 NumParamFree++;
			 if(NumParamFree>1) FoundParamNotZero = true;
//     			 coeffPoly.emplace_back(varName, parIni,0.00001,0.,1000.);
//			  coeffPoly.emplace_back(varName, parIni,-1*NParIni*fabs(parIni),NParIni*fabs(parIni));
//			  coeffPoly.emplace_back(varName, parIni,fabs(parIni)/1000.,-1*NParIni*fabs(parIni),NParIni*fabs(parIni));
			 }else{
       			  coeffPoly.emplace_back(varName, 0.);
//       			  coeffPoly.emplace_back(varName, parIni,-1.0,1.0);
			 } 
			 if(i==fixParam && parIni!=1.00000){
     std::cout<<"====================================================================================="<<std::endl;
     std::cout<<"====================================================================================="<<std::endl;
     std::cout<<"====================================================================================="<<std::endl;
     std::cout<<Form("Error reading parameters file: fixparam=%d  %s=%f !=1.00 !!!",fixParam,varName,parIni)<<std::endl;
     std::cout<<"====> Try to search fixparam!!!"<<std::endl;
//     std::cout<<"====> EXIT from Fit!!!"<<std::endl;
     std::cout<<"====================================================================================="<<std::endl;
     std::cout<<"====================================================================================="<<std::endl;
     std::cout<<"====================================================================================="<<std::endl;
     //exit(0);
     SearchFixParam = true;
			 }
			 if(SearchFixParam&&fixParamTmp>=0){
			  fixParam = fixParamTmp;
			  if(i==fixParam){
     std::cout<<"====================================================================================="<<std::endl;
     std::cout<<"====================================================================================="<<std::endl;
     std::cout<<Form("Search FixParam Found: fixparam=%d  %s=%f =1.00 !!!",fixParam,varName,parIni)<<std::endl;
     std::cout<<"====================================================================================="<<std::endl;
     std::cout<<"====================================================================================="<<std::endl;
			  }
			 }
     }
    if(!FoundParamNotZero){
     std::cout<<"====================================================================================="<<std::endl;
     std::cout<<"====================================================================================="<<std::endl;
     std::cout<<"====================================================================================="<<std::endl;
     std::cout<<"Error reading parameters file: not found any free parameter different from 0.0 !!!!"<<std::endl;
     std::cout<<"====> EXIT from Fit!!!"<<std::endl;
     std::cout<<"====================================================================================="<<std::endl;
     std::cout<<"====================================================================================="<<std::endl;
     std::cout<<"====================================================================================="<<std::endl;
     exit(0);
    } 
    parListInput->close(); 
    sprintf(testo,"cp %s %s.tmp",ListParName,ListParName);
    gSystem->Exec(testo);
//     sprintf(testo,"cp %s %s.tmp",ListParNorm,ListParNorm);
//     gSystem->Exec(testo);
    parListInput->clear(); 
    sprintf(testo,"cp %s %s.tmp",ListPloName,ListPloName);
    gSystem->Exec(testo);
//     sprintf(testo,"cp %s %s.tmp",ListPloNorm,ListPloNorm);
//     gSystem->Exec(testo);
  }else{
     std::cout<<"First FIT: let's search the parameter to fix... "<<std::endl;
     FirstFit=true;
     TRandom3* trand_ini = new TRandom3(time(0));
     std::cout<<"List of initial parameters "<< ListParName <<" not found"<<std::endl;
     if(Q2Bin==4) {
      NumCalls = 1200000;
      boolHesse = false;
     }else{
      NumCalls = 200000;
      boolHesse = true;
     } 
     FitPrintLevel=0;
     for (int i=0;i<numParameters;++i){
            if(i==fixParam){
//            if(i==fixParam&&!AutoFixPar){
	     parIni=1.;
             std::cout<<"Warning !!!! Setting p"<<fixParam<<"=1 [because the normalization condition of PDF the (free num par)=(num par-1)"<<std::endl;
	    }else{ 
             parIni = trand_ini->Uniform(RndMin,RndMax);
	     NumParamFree++;
	    } 
     	    sprintf(varName, "p%d", i);
// 	    if(i>=24&&i<=47&&i!=44){
//      			 coeffPoly.emplace_back(varName, 0.0000,0.001,ParMin,ParMax);
// 			 
// 	    }else{
     			 coeffPoly.emplace_back(varName, parIni,0.001,ParMin,ParMax);
//	    }		 

     }
  }   
   
     int   initCoeffFit = limits.size();
//     GooFit::BernsteinPdf    *model=0;
//  if(integral){
         GooFit::FastBernsteinPdf    *model     =  new GooFit::FastBernsteinPdf("model",obsPoly,coeffPoly,limits,maxDegree1,maxDegree2,maxDegree3);
//  }else{
//        model     =  new GooFit::BernsteinPdf("model",obsPoly,coeffPoly,limits,maxDegree1,maxDegree2,maxDegree3);
//  } 

//================================================================================
//================================================================================
///FIT
//================================================================================
//================================================================================


  model->setData(dataReco);

//  GooFit::FitManager fitter(&model);//
//  int NumCalls = 1500000;

  if(SetMinuit2){
   GooFit::FitManagerMinuit2 fitter(model);
   fitter.setMaxCalls(NumCalls);
   fitter.setVerbosity(2);
   fitter.fit();
  }else{
      std::cout<<"Warning !!!! bSetting num call for MINUIT :"<<NumCalls  <<std::endl;
      GooFit::FitManagerMinuit1 fitter(model);
      fitter.setMaxCalls(NumCalls);
      fitter.useHesseBefore(false);
      fitter.useHesse(boolHesse);
      fitter.useMinos(false);
      cout<<"\n"<<endl;
      cout<<"		       ===*** Start Fit ***=== "<<endl;
      cout<<"		       ===*** Start Fit ***=== "<<endl;
      cout<<"		       ===*** Start Fit ***=== "<<endl;
      cout<<"\n"<<endl;

      Minuit = fitter.getMinuitObject();
      Minuit->SetPrintLevel(FitPrintLevel);
//        Minuit->SetErrorDef(1.);
//     //  Minuit->SetErrorDef(0.5);
//      double arglist[2];
//      int err = 0;
//      arglist[0]= 120000; // maximum iterations
//      Minuit->Migrad();

// AutoFixPar
//
      if(AutoFixPar&&FirstFit){
      
        int SBEntriesDataReco=dataReco->getNumEvents();

//        CutSignificance=3.;  
        //CutSignificance=(sqrt(SBEntriesDataReco/50.));  
//        CutSignificance=round(sqrt(SBEntriesDataReco/50.));  
//        CutSignificance=round(sqrt(dataReco->getNumEvents()/50));  
        std::cout<<"----------------------------------"<<std::endl ;
        std::cout<<Form("--> Setting Significance CUT = %f SBEvents=%d [%f] <--",CutSignificance,SBEntriesDataReco,sqrt(dataReco->getNumEvents()/50.))<<std::endl ;
        std::cout<<"----------------------------------"<<std::endl ;
//
        std::cout<<"=================================="<<std::endl ;
        std::cout<<"=================================="<<std::endl ;
        std::cout<<"==> Begin First FIT   <=="<<std::endl ;
        std::cout<<"=================================="<<std::endl ;
        std::cout<<"=================================="<<std::endl ;
//        double CutParValue0 = 0.000000001;
//        double CutParValue = 0.01;
	double CutParValue     = 0.01;
//	double CutParValue     = 0.000000001;
//	double CutParValueNorm = 0.01;
//	double CutParValueLast = 0.12;
        //double CutParValue = 0.01;
        //double CutParValueLast = 0.20;
//        double CutParValue = 0.09999;
//	double SigmaCut1 =1.00;
//	double SigmaCut2 =2;
	
//	if(NFactGen==1) CutSignificance=round(sqrt(dataReco->getNumEvents()/200)*CutSignificance);  
//	double SigmaCut2 =2.5*round(sqrt(NFactGen));
	double SigmaCut2 =CutSignificance*round(sqrt(NFactGen));
//	double SigmaCut2 =CutSignificance;
	if (SigmaCut2>5) SigmaCut2=5;
	
	
//	double SigmaCut25 =CutSignificance;
	double SigmaCut25 =CutSignificance*round(sqrt(NFactGen));
	if (SigmaCut25>5) SigmaCut25=5;
	
//	double SigmaCut2 =2.00*(sqrt(HxReco->GetEntries()/257));
	int err=1;
        double XStep=0.0001;
	bool IlFix= false;
	int IlFirstFix = initCoeffFit+fixParam;
	if (fixParam<=numParameters){
	 IlFix= true;
 	 std::cout<<"===================================================================="<<std::endl ;
	 std::cout<<Form("Parameter for NORMALIZATION IS SET fix to  ==> p(%d)=1!!!!",fixParam)<<std::endl;
 	 std::cout<<"===================================================================="<<std::endl ;
	}  
	std::cout<<Form("=================== [Zero FIT Loop Start] ===================\n")<<std::endl; 
//=====================================================================
//=====================================================================
//  	Minuit->mnrset(1);
//   	 for (int i=0;i<numParameters;++i){
// 	       if (IlFix&&i==fixParam) continue;
// 	        Minuit->Release(i+initCoeffFit);
//  	        Minuit->mnparm(i+initCoeffFit, varName,coeffPoly[i].getValue() ,XStep,ParMin,ParMax,err );
// 	 }      
  	if(IlFix) Minuit->FixParameter(IlFirstFix);
//  	for (int i=0;i<numParameters;++i){
// 	   if(coeffPoly[i].getValue()==0.){
//      	    sprintf(varName, "p%d", i);
//  	    Minuit->mnparm(i+initCoeffFit, varName, 0.,XStep, -1.,1.,err);
//  	    Minuit->FixParameter(i+initCoeffFit);
// 	   } 
//         }
   	fitter.fit();
//First loop 
//        NumCalls=150000;	
//        fitter.setMaxCalls(NumCalls);
	bool FitAgain=true;
	std::cout<<Form("=================== [First FIT Loop Start] ===================\n")<<std::endl; 
	for (int iLoop=0;iLoop<numParameters;++iLoop){
//        double min_val =999;
//	int il_val =-999;
	 FitAgain=false;
 	 for (int i=0;i<numParameters;++i){
	   if(coeffPoly[i].getError()==0.) continue;
 	   sprintf(varName, "p%d", i);
	   double coeff_sign = coeffPoly[i].getValue()/fabs(coeffPoly[i].getError());
//           double coeff_val = coeffPoly[i].getValue();
// 	   if(fabs(coeffPoly[i].getError())!=0.&& (coeff_sign<1.&&coeffPoly[i].getValue()>1. || fabs(coeffPoly[i].getValue())<CutParValue&&coeffPoly[i].getValue()<1.) ){
// 	   if(coeff_val<min_val){
// 	    min_val=coeff_val;
// 	    il_val=i;
// 	    FitAgain=true;
// 	    std::cout<<Form("VALMINIMUM = %f IPar=%d",min_val,il_val)<<std::endl;
// 	   } 
	   if(fabs(coeffPoly[i].getValue())<CutParValue&&coeffPoly[i].getError()!=0.) {
	    std::cout<<Form("==>[First FIT Loop] Force setting p(%d)=0  [value=%3.10f<%f] [significance=%3.10f]",i,coeffPoly[i].getValue(),CutParValue,coeff_sign)<<std::endl;
	    FitAgain=true;
	    coeffPoly[i] =0.00;
 	    Minuit->mnparm(i+initCoeffFit, varName, 0.,XStep, -1.,1.,err);
 	    Minuit->FixParameter(i+initCoeffFit);
// 	   }else if(fabs(coeffPoly[i].getValue())<0.0001&&coeffPoly[i].getError()!=0.&&iLoop==0) {
// 	    std::cout<<Form("==>[First FIT Loop] Force setting p(%d)=0  [value=%f<%f] [significance=%f]",i,coeffPoly[i].getValue(),CutParValue,coeff_sign)<<std::endl;
// 	    FitAgain=true;
// 	    coeffPoly[i] =0.00;
//  	    Minuit->mnparm(i+initCoeffFit, varName, 0.,XStep, -1.,1.,err);
//  	    Minuit->FixParameter(i+initCoeffFit);
 	   }else{
  	    Minuit->mnparm(i+initCoeffFit, varName, coeffPoly[i].getValue(),XStep,ParMin,ParMax,err);
	   }
 	 } 
// 	 if (il_val>=0&&min_val<CutParValue&&coeffPoly[il_val].getError()!=0.000){
// 	  std::cout<<Form("==>[First Loop=%d] Force setting p(%d)=0  [value=%3.12f<%f] \
// 	  [minimum value=%3.12f]",iLoop,il_val,coeffPoly[il_val].getValue(),CutParValue,min_val)<<std::endl;
// 	  coeffPoly[il_val] =0.00;
//  	  sprintf(varName, "p%d", il_val);
//  	  Minuit->mnparm(il_val+initCoeffFit, varName, 0.,XStep,-1.,1.,err);
//  	  Minuit->FixParameter(il_val+initCoeffFit);
// 	 }else{
// 	  FitAgain=false;
// 	 }
	 Minuit->mnrset(1);
 	 fitter.fit();
  	 if(!FitAgain) break;
 	 std::cout<<"=================================="<<std::endl ;
 	 std::cout<<Form("==> Begin FIT %d",iLoop)<<std::endl ;
 	 std::cout<<"=================================="<<std::endl ;
//	 Minuit->mnrset(1);
	}
// Exit from loop, search the param with greater significance
//        ParMax=1.;
        if(Q2Bin==4){
          fitter.setMaxCalls(NumCalls);
          fitter.useHesse(true);
	}  
	if(!IlFix){
 	 std::cout<<"=================================="<<std::endl ;
 	 std::cout<<Form("==> EXIT FROM LOOP, search the param")<<std::endl ;
	 std::cout<<Form("with greater significance")<<std::endl ;
 	 std::cout<<"=================================="<<std::endl ;
         Minuit->Release(IlFirstFix);
 	 for(int i=0;i<numParameters;++i) {
	  if(coeffPoly[i].getError()==0.) continue;
	  double coeff_sign = coeffPoly[i].getValue()/fabs(coeffPoly[i].getError());
 	   std::cout<<"=================================="<<std::endl ;
 	   std::cout<<Form("Par Value = %f Err = %f Signif = %f",coeffPoly[i].getValue(),coeffPoly[i].getError(),coeff_sign)<<std::endl ;
 	   std::cout<<"=================================="<<std::endl ;
//  	  if(fabs(coeffPoly[i].getValue())<CutParValue&&coeffPoly[i].getError()!=0.){
// 	   coeffPoly[i] =0.00;
// 	   std::cout<<Form("==>[search max significance] Force setting p(%d)=0 [value=%f<%f] significance=%f",i,coeffPoly[i].getValue(),CutParValue,coeff_sign)<<std::endl;
// 	  } 
//	  if(fabs(coeffPoly[i].getValue())<coeffPoly[i].getError()) coeffPoly[i] =0.00;
//ok 	  if(fabs(coeffPoly[i].getValue())<CutParValue&&coeffPoly[i].getError()!=0.) coeffPoly[i] =0.00;
// 	  if(fabs(coeffPoly[i].getError())!=0.&& (coeff_sign<1.&&coeffPoly[i].getValue()>1. || fabs(coeffPoly[i].getValue())<CutParValue&&coeffPoly[i].getValue()<1.) ) coeffPoly[i] =0.00;
  	  if(fabs(coeffPoly[i].getValue())>1.&&coeffPoly[i].getError()>0. ){
 	   double xCoeffNormTmp = fabs(coeffPoly[i].getValue()/coeffPoly[i].getError());
 	   if (xCoeffNormTmp>xCoeffNorm) {
 	    xCoeffNorm=xCoeffNormTmp;
 	    xCoeffIndex=i;
 	   }
 	  }
	 }
	  if(xCoeffIndex<0&&fixParam>=numParameters){
 	   std::cout<<"=================================="<<std::endl ;
 	   std::cout<<"Parameter for NORMALIZATION not found!!! Exit..."<<std::endl ;
 	   std::cout<<"=================================="<<std::endl ;
	   exit(1);
	  }else if(xCoeffIndex<0&&fixParam<numParameters){
	   xCoeffIndex=fixParam;
 	   std::cout<<"=================================="<<std::endl ;
	   std::cout<<Form("Parameter for NORMALIZATION remain fix to  ==> p(%d)=1!!!!",xCoeffIndex)<<std::endl;
 	   std::cout<<"=================================="<<std::endl ;
	  }else{
 	   std::cout<<"=================================="<<std::endl ;
	   std::cout<<Form("Parameter for NORMALIZATION is set by loop to ==> p(%d)=1!!!!",xCoeffIndex)<<std::endl;
 	   std::cout<<"=================================="<<std::endl ;
	  }
//       TRandom3* trand_ini2 = new TRandom3();
 	 for(int i=0;i<numParameters;++i) {
	  if(coeffPoly[i].getError()==0.) continue;
 	  sprintf(varName, "p%d", i);
	  if(coeffPoly[i].getValue()<=0.0){
 	     Minuit->mnparm(i+initCoeffFit, varName, 0.,XStep, -1.,1.,err);
 	     Minuit->FixParameter(i+initCoeffFit);
	  }
	  if(coeffPoly[i].getValue()>0.0&&i!=xCoeffIndex){
	    double parRenorm = coeffPoly[i].getValue()/coeffPoly[xCoeffIndex].getValue();
 	    Minuit->mnparm(i+initCoeffFit, varName,parRenorm,XStep, ParMin,ParMax,err);
	  }
//	    double parIni2 = trand_ini2->Uniform(RndMin,RndMax);
//	 Minuit->mnparm(i+initCoeffFit, varName,parIni2,0.001, ParMin,ParMax,err);
	 }
  	 sprintf(varName, "p%d",xCoeffIndex );
 	 Minuit->mnparm(xCoeffIndex+initCoeffFit, varName,1.,XStep, -1.,1.,err);
 	 Minuit->FixParameter(xCoeffIndex+initCoeffFit);
	 Minuit->mnrset(1);
 	 fitter.fit();
	}else{
	   xCoeffIndex=fixParam;
 	   std::cout<<"=================================="<<std::endl ;
	   std::cout<<Form("Parameter for NORMALIZATION REMAIN SET to ==> p(%d)=1!!!!",xCoeffIndex)<<std::endl;
 	   std::cout<<"=================================="<<std::endl ;
	}
 	   std::cout<<"=================================="<<std::endl ;
	   std::cout<<"         DUMP SIGNIFICANCES"<<std::endl;
 	   std::cout<<"=================================="<<std::endl ;
 	for (int i=0;i<numParameters;++i){
 	  if(coeffPoly[i].getError()==0.000&&coeffPoly[i].getValue()==0.000) continue;
 	  double coeff_sign = coeffPoly[i].getValue()/fabs(coeffPoly[i].getError());
	  if(coeff_sign>1){
	  std::cout<<Form("==>CHECKSIGN Ok significance p(%d)=0 [value=%f] [cut significance %f>1 (%f)] \
	  ",i,coeffPoly[i].getValue(),coeff_sign,SigmaCut2)<<std::endl;
	  }else{
	  std::cout<<Form("==>CHECKSIGN CUT significance p(%d)=0 [value=%f] [cut significance %f<1 (%f)] \
	  ",i,coeffPoly[i].getValue(),coeff_sign,SigmaCut2)<<std::endl;
	  }
	} 
// start a loop to fix the model...	
//        ParMax=1;

        if (Q2Bin==4) {
	  fitter.useHesse(true);
	  std::cout<<Form("Q2Bin=4 => Set fitter.useHesse(true)")<<std::endl;
	 } 
        double min_sign =999;
	int il_sign =-999;
	for (int iLoop=0;iLoop<numParameters;++iLoop){
//	 cout<<"=========================="<<endl;
//	 cout<<Form("==> FIX Loop N.%d",iLoop)<<endl;
	 NumParamFree=1.;
         min_sign =999;
	 il_sign =-999;
	 FitAgain=false;
 	 for (int i=0;i<numParameters;++i){
 	   if(coeffPoly[i].getError()==0.000&&coeffPoly[i].getValue()==0.000) continue;
 	   if(coeffPoly[i].getError()==0.000&&coeffPoly[i].getValue()==1.000) continue;
// 	   if(fabs(coeffPoly[i].getValue())<CutParValue&&coeffPoly[i].getError()!=0.) {
 	   double coeff_sign = coeffPoly[i].getValue()/fabs(coeffPoly[i].getError());
	   if(coeff_sign<min_sign){
	    min_sign=coeff_sign;
	    il_sign=i;
	    FitAgain=true;
	    std::cout<<Form("MINIMUM = %f IPar=%d loop=%d",min_sign,il_sign,iLoop)<<std::endl;
	   } 
// 	 if(fabs(coeffPoly[i].getError())!=0.&&  fabs(coeffPoly[i].getValue())<CutParValue&&coeff_sign<1 ) {
//ok 	 if(fabs(coeffPoly[i].getError())!=0.&& (coeff_sign<1.&&coeffPoly[i].getValue()>1. || fabs(coeffPoly[i].getValue())<CutParValue&&coeffPoly[i].getValue()<1.) ) {
// 	 if(fabs(coeffPoly[i].getError())!=0.&& (coeff_sign<1.&&coeffPoly[i].getValue()>1. || fabs(coeffPoly[i].getValue())<CutParValue&&coeffPoly[i].getValue()<1.) ) {
//  	 if(fabs(coeffPoly[i].getError())!=0.&&  fabs(coeffPoly[i].getValue())<CutParValue ) {
// 	  std::cout<<Form("==>[Loop=%d to fix model]  Force setting p(%d)=0 [value=%f<%f] significance=%f",iLoop,i,coeffPoly[i].getValue(),CutParValue,coeff_sign)<<std::endl;
// 	    FitAgain=true;
// 	    coeffPoly[i] =0.00;
//  	    Minuit->mnparm(i+initCoeffFit, varName, 0.,XStep,-1.,1.,err);
//  	    Minuit->FixParameter(i+initCoeffFit);
// 	 }
// 	 if((fabs(coeffPoly[i].getValue())<fabs(coeffPoly[i].getError())||fabs(coeffPoly[i].getValue())<CutParValue)&&iLoop>1) {
//ref 	 if(fabs(coeffPoly[i].getValue())<fabs(coeffPoly[i].getError())&&iLoop>1) {
//04062020
//  	   if(coeff_sign<SigmaCut1&&iLoop==iFitLoop) {
// 	    std::cout<<Form("==>[Loop=%d to fix model] Force setting p(%d)=0  [value=%f] \
// 	    [significance=%f<%f]",iLoop,i,coeffPoly[i].getValue(),coeff_sign,SigmaCut1)<<std::endl;
// 	    FitAgain=true;
// 	    coeffPoly[i] =0.00;
//  	    Minuit->mnparm(i+initCoeffFit, varName, 0.,XStep,-1.,1.,err);
//  	    Minuit->FixParameter(i+initCoeffFit);
// //	   }else{
// //	    Minuit->mnparm(i+initCoeffFit, varName, coeffPoly[i].getValue(),0.001,ParMin,ParMax,err);
// 	   }
	   if(coeffPoly[i].getValue()>0.){
	    NumParamFree++;
	   } ;
	 }// end loop params 
 	 std::cout<<"=================================="<<std::endl ;
 	 std::cout<<Form("==> Begin FIT Step %d Fixing par%d=1",iLoop,xCoeffIndex)<<std::endl ;
 	 std::cout<<"=================================="<<std::endl ;
//	 if (il_sign>=0&&min_sign<SigmaCut2){
	 if (il_sign>=0&&min_sign<SigmaCut2&&coeffPoly[il_sign].getError()!=0.000){
	  std::cout<<Form("==>[Loop=%d to fix model] Force setting p(%d)=0 [value=%f] [minimum significance=%f<%f] \
	  ",iLoop,il_sign,coeffPoly[il_sign].getValue(),min_sign,SigmaCut2)<<std::endl;
	  coeffPoly[il_sign] =0.00;
 	  sprintf(varName, "p%d", il_sign);
 	  Minuit->mnparm(il_sign+initCoeffFit, varName, 0.,XStep,-1.,1.,err);
 	  Minuit->FixParameter(il_sign+initCoeffFit);
	 }else{
	  FitAgain=false;
          Minuit->SetPrintLevel(2);
	 }
//  	 for (int i=0;i<numParameters;++i){
// 	  if(coeffPoly[i].getError()==0.000&&coeffPoly[i].getValue()==0.000) continue;
// 	  if(coeffPoly[i].getError()==0.000&&coeffPoly[i].getValue()==1.000) continue;
// 	  if(i==il_sign) continue;
// 	  if(i==xCoeffIndex) continue;
//  	  sprintf(varName, "p%d", i);
//   	  Minuit->mnparm(i+initCoeffFit, varName, 0.1,XStep,ParMin,ParMax,err);
// 	 } 
	 Minuit->mnrset(1);
 	 fitter.fit();
  	 if(!FitAgain){
//  	  NumParamFree=1.;
//    	  for (int i=0;i<numParameters;++i){
//  	   if(coeffPoly[i].getError()==0.000&&coeffPoly[i].getValue()==0.000) continue;
//  	   if(coeffPoly[i].getError()==0.000&&coeffPoly[i].getValue()==1.000) continue;
// 	 
//  	   double coeff_sign = coeffPoly[i].getValue()/fabs(coeffPoly[i].getError());
// 	   if(coeffPoly[i].getValue()<CutParValueLast&&coeff_sign<SigmaCut25) {
// 	    std::cout<<Form("==>[last fit] Force setting p(%d)=0 [value=%f<%f] \
// 	    [significance=%f<%f]",i,coeffPoly[i].getValue(),CutParValueLast,coeff_sign,SigmaCut25)<<std::endl;
// 	    coeffPoly[i] =0.00;
// 	    Minuit->mnparm(i+initCoeffFit, varName, 0.,XStep,-1.,1.,err);
// 	    Minuit->FixParameter(i+initCoeffFit);
// 	   } 
// 	   if(coeffPoly[i].getValue()>0.){
// 	    NumParamFree++;
//  	   } ;
// 	  } 
//  	  Minuit->mnrset(1);
//  	  fitter.fit();
	  if(!IlFix){
 	   xCoeffNorm =0.0;
 	   int xCoeffIndexTmp=-1;
 	   for(int i=0;i<numParameters;++i) {
 	    if((coeffPoly[i].getValue())>1. && fabs(coeffPoly[i].getError()>0.) ){
// 	    if((coeffPoly[i].getValue()+coeffPoly[i].getError())>1. && fabs(coeffPoly[i].getError()>0. && i<xCoeffIndex) ){
 	     double xCoeffNormTmp = coeffPoly[i].getValue();
//	     double xCoeffNormTmp = coeffPoly[i].getValue()/fabs(coeffPoly[i].getError());
 	     if (xCoeffNormTmp>xCoeffNorm) {
 	      xCoeffNorm=xCoeffNormTmp;
 	      xCoeffIndexTmp=i;
 	     }
 	    }
 	   }
	   if( xCoeffIndexTmp>0 ){
	    xCoeffIndex=xCoeffIndexTmp;
 	    std::cout<<"=============================================================="<<std::endl;
 	    std::cout<<"=============================================================="<<std::endl;
	    std::cout<<Form("WARNING: Normalization could be fixed better if p(%d)=1",xCoeffIndex)<<std::endl ;
	    std::cout<<Form("WARNING: Try to fit fixing p(%d)=1",xCoeffIndex)<<std::endl ;
 	    std::cout<<"=============================================================="<<std::endl;
 	    std::cout<<"=============================================================="<<std::endl;
 	    for(int i=0;i<numParameters;++i) {
	     if(coeffPoly[i].getValue()>0.000) {
 	      Minuit->Release(i+initCoeffFit);
 	      double coeff_sign = coeffPoly[i].getValue()/fabs(coeffPoly[i].getError());
	      double parRenorm = coeffPoly[i].getValue()/xCoeffNorm;
 	      sprintf(varName, "p%d", i);
 	       Minuit->mnparm(i+initCoeffFit, varName,parRenorm,XStep, ParMin,ParMax,err);
 	     }
	    }
	    Minuit->FixParameter(xCoeffIndex+initCoeffFit);
	    Minuit->mnrset(1);
 	    fitter.fit();
	    FitAgain=true;
	   }
 	  }
	  if(!FitAgain){
 	   std::cout<<Form("==> BREAK: EXIT FIT Step %d",iLoop)<<std::endl ;
	   break;
	  } 
	 } 
	}
        std::cout<<"Count NumParamFree "<<NumParamFree<<std::endl;
	if(NumParamFree<=1){
         std::cout<<"====> Error!!! Exit... "<<std::endl;
	 exit(0);
	}
/* 	if(!IlFix){
 	 xCoeffNorm =0.0;
 	 int xCoeffIndexTmp=-1;
 	 for(int i=0;i<numParameters;++i) {
 	  if(fabs(coeffPoly[i].getValue())>1. && fabs(coeffPoly[i].getError()>0.) ){
 	   double xCoeffNormTmp = coeffPoly[i].getValue();
//	   double xCoeffNormTmp = coeffPoly[i].getValue()/fabs(coeffPoly[i].getError());
 	   if (xCoeffNormTmp>xCoeffNorm) {
 	    xCoeffNorm=xCoeffNormTmp;
 	    xCoeffIndexTmp=i;
 	   }
 	  }
 	 }
	 if( xCoeffIndexTmp>0 ){
	  xCoeffIndex=xCoeffIndexTmp;
 	  std::cout<<"=============================================================="<<std::endl;
 	  std::cout<<"=============================================================="<<std::endl;
	  std::cout<<Form("WARNING: Normalization could be fixed better if p(%d)=1",xCoeffIndex)<<std::endl ;
	  std::cout<<Form("WARNING: Try to fit fixing p(%d)=1",xCoeffIndex)<<std::endl ;
 	  std::cout<<"=============================================================="<<std::endl;
 	  std::cout<<"=============================================================="<<std::endl;
 	  for(int i=0;i<numParameters;++i) {
	   if(coeffPoly[i].getValue()>0.000) {
            Minuit->Release(i+initCoeffFit);
  	    double coeff_sign = coeffPoly[i].getValue()/fabs(coeffPoly[i].getError());
	    double parRenorm = coeffPoly[i].getValue()/xCoeffNorm;
            sprintf(varName, "p%d", i);
// 	    if (parRenorm>CutParValueNorm){
   	     Minuit->mnparm(i+initCoeffFit, varName,parRenorm,XStep, ParMin,ParMax,err);
// 	    }else{
// 	     std::cout<<Form("==>[try last renormalisation] Force setting p(%d)=0 [significance=%f]",i,coeff_sign)<<std::endl;
// 	     Minuit->mnparm(i+initCoeffFit, varName,0.000,0.001, ParMin,ParMax,err);
//             Minuit->FixParameter(i+initCoeffFit);
// 	    }
   	   }
	  } 
	  Minuit->FixParameter(xCoeffIndex+initCoeffFit);
	  Minuit->mnrset(1);
 	  fitter.fit();
	 } 
        }
 *///        Minuit->mnrset(1);
/*        std::cout<<"=========================="<<std::endl;
       std::cout<<"=       LAST FIT!!!      ="<<std::endl;
       std::cout<<"=========================="<<std::endl;
       for (int iLoop=0;iLoop<numParameters;++iLoop){
 	NumParamFree=1.;
	FitAgain=false;
   	for (int i=0;i<numParameters;++i){
 	 if(coeffPoly[i].getError()==0.000&&coeffPoly[i].getValue()==0.000) continue;
 	 if(coeffPoly[i].getError()==0.000&&coeffPoly[i].getValue()==1.000) continue;
	
 	 double coeff_sign = coeffPoly[i].getValue()/fabs(coeffPoly[i].getError());
 	 double valmenosigma  = coeffPoly[i].getValue()-1.*fabs(coeffPoly[i].getError());
	 std::cout<<Form("==> coeffPoly[%d] valmenosigma=%f coeff_sign=%f\n",i,valmenosigma,coeff_sign);
//	 if((coeffPoly[i].getValue()-3.*coeffPoly[i].getError())<CutParValueLast&&coeff_sign<SigmaCut25) {
// 	 if(valmenosigma<CutParValueLast&&coeff_sign<SigmaCut25) {
//           sprintf(varName, "p%d", i);
// 	  std::cout<<Form("==>[LAST FIT Loop=%d] Force setting p(%d)=0 [value=%f<%f] \
// 	  [significance=%f<%f]",iLoop,i,coeffPoly[i].getValue(),CutParValueLast,coeff_sign,SigmaCut25)<<std::endl;
// 	  coeffPoly[i] =0.00;
// 	  Minuit->mnparm(i+initCoeffFit, varName, 0.,XStep,-1.,1.,err);
// 	  Minuit->FixParameter(i+initCoeffFit);
// 	  FitAgain=true;
// 	 } 
	 if (coeff_sign<SigmaCut25&&coeffPoly[i].getError()!=0.000){
          sprintf(varName, "p%d", i);
	  std::cout<<Form("==>[LAST FIT Loop=%d] Force setting p(%d)=0 [value=%f] [minimum significance=%f<%f] \
	  ",iLoop,i,coeffPoly[i].getValue(),coeff_sign,SigmaCut2)<<std::endl;
	  coeffPoly[i] =0.00;
	  Minuit->mnparm(i+initCoeffFit, varName, 0.,XStep,-1.,1.,err);
	  Minuit->FixParameter(i+initCoeffFit);
	  FitAgain=true;
         }
	 if(coeffPoly[i].getValue()>0.){
	  NumParamFree++;
 	 } ;
	} 
 	Minuit->mnrset(1);
 	fitter.fit();
        if(!FitAgain){
 	  std::cout<<Form("==> BREAK: EXIT LAST FIT Step %d",iLoop)<<std::endl ;
	  break;
	}  
       }	
 */
// 	for (int iLoop=0;iLoop<=iFitLoop+1;++iLoop){
// 	 cout<<"=========================="<<endl;
// 	 cout<<Form("==> LAST Loop N.%d",iLoop)<<endl;
//  	 for (int i=0;i<numParameters;++i){
//  	  if(coeffPoly[i].getValue()==0.&&coeffPoly[i].getError()==0.) continue;
//  	  sprintf(varName, "p%d", i);
// //	  if(fabs(coeffPoly[i].getValue())<CutParValue&&coeffPoly[i].getError()!=0.) {
//  	  double coeff_sign = coeffPoly[i].getValue()/fabs(coeffPoly[i].getError());
// 	   
// 	  Minuit->mnparm(i+initCoeffFit, varName, coeffPoly[i].getValue(),0.001,ParMin,ParMax,err);
// 	  
// //  	  if(coeff_sign<SigmaCut2&&iLoop>iFitLoop ) {
// // //	  if(coeff_sign<round(sqrt(NFactGen))&&iLoop==1) {
// // 	   std::cout<<Form("==>[last fit] Force setting p(%d)=0 [value=%f]\
// // 	   [significance=%f<%f]",i,coeffPoly[i].getValue(),coeff_sign,SigmaCut2)<<std::endl;
// // 	   coeffPoly[i] =0.00;
// //  	   Minuit->mnparm(i+initCoeffFit, varName, 0.,XStep,-1.,1.,err);
// //  	   Minuit->FixParameter(i+initCoeffFit);
// /* // 	  }
// */ 	  if(coeff_sign<SigmaCut2&&iLoop>=iFitLoop ) {
// 	   std::cout<<Form("==>[last fit] Force setting p(%d)=0 [value=%f]\
// 	   [significance=%f<%f]",i,coeffPoly[i].getValue(),coeff_sign,SigmaCut2)<<std::endl;
// 	   coeffPoly[i] =0.00;
//  	   Minuit->mnparm(i+initCoeffFit, varName, 0.,XStep,-1.,1.,err);
//  	   Minuit->FixParameter(i+initCoeffFit);
// 	  }
// //	  if(coeffPoly[i].getValue()+SigmaCut1*coeffPoly[i].getError()<CutParValueLast) {
// 	  if(coeffPoly[i].getValue()<CutParValueLast) {
// 	   std::cout<<Form("==>[last fit] Force setting p(%d)=0 [value=%f<%f] significance=%f",i,coeffPoly[i].getValue(),CutParValueLast,coeff_sign)<<std::endl;
// 	   coeffPoly[i] =0.00;
// 	   Minuit->mnparm(i+initCoeffFit, varName, 0.,XStep,-1.,1.,err);
// 	   Minuit->FixParameter(i+initCoeffFit);
// 	  } 
//  	 }// end loop on params 
// 	 
// 	 Minuit->FixParameter(xCoeffIndex+initCoeffFit);
//  	 fitter.fit();
// 	}// end loop on fits
// 	 
	NumParamFree=0; 
 	for (int i=0;i<numParameters;++i){
	 if(coeffPoly[i].getValue()>0.){
	  NumParamFree++;
	 } ;
        } ;
        std::cout<<"After fit - Count NumParamFree "<<NumParamFree<<std::endl;
	if(NumParamFree<=1){
         std::cout<<"====> Error!!! Exit... "<<std::endl;
	 exit(0);
	} 
  	std::cout<<"=============================================================="<<std::endl;
 	std::cout<<"=============================================================="<<std::endl;
 	std::cout<<Form("REMIND: In This Fit Normalization parameter is ==> p(%d)=1!!!!",xCoeffIndex)<<std::endl;
 	std::cout<<"=============================================================="<<std::endl;
 	std::cout<<"=============================================================="<<std::endl;
 	std::cout<<"=============================================================="<<std::endl;
      }else{
//
//  Manual Fit
//
  	std::cout<<"=============================================================="<<std::endl;
 	std::cout<<"=============================================================="<<std::endl;
 	std::cout<<Form("                      START MANUAL FIT")<<std::endl;
 	std::cout<<"=============================================================="<<std::endl;
 	std::cout<<"=============================================================="<<std::endl;
 	std::cout<<"=============================================================="<<std::endl;
        Minuit->FixParameter(initCoeffFit+fixParam);
        std::cout<<"Warning !!!! Fixing p"<<fixParam<<"=1"<<std::endl;
//          Minuit->FixParameter(19);
        for (int i=0;i<numParameters;++i){
          if (coeffPoly[i].getValue()==0.0){
            Minuit->FixParameter(i+initCoeffFit);
          }
         }
         std::fstream *parListFix = new std::fstream(FitStraName,std::ifstream::in);
         if(parListFix->is_open()){
          parListFix->close();
          std::cout<<"==========================================================================="<<std::endl ;
          std::cout<<"Namelist: "<<FitStraName<<" to fix parameters exist! Opening..."<<std::endl ;
          std::cout<<"==========================================================================="<<std::endl ;
          char*argm[]={FitStraName};
          std::map<std::string, std::string> mapfix = ReadNamelist(1,argm );
          for (int i=0;i<numParameters;++i){
            if( strcmp(mapfix[coeffPoly[i].getName()].c_str(),"fix")>=0){
             Minuit->FixParameter(i+initCoeffFit);
             std::cout<<"==========================================================================="<<std::endl ;
             std::cout<<"Fixing Parameter -> "<<coeffPoly[i].getName()<<std::endl;
             std::cout<<"==========================================================================="<<std::endl ;
            }
          }
         }else{
           std::cout<<"==========================================================================="<<std::endl ;
           std::cout<<"Warning ! Namelist: "<<FitStraName<<" to fix parameters in the fit doesn't exist. Proceed ahead..."<<std::endl ;
           std::cout<<"==========================================================================="<<std::endl ;
         }
//      Minuit->FixParameter(10);
        fitter.fit();
     }
//
// save covariance Matrix     	
// 
     Double_t matrix[NumParamFree-1][NumParamFree-1];
     Minuit->mnemat(&matrix[0][0],NumParamFree-1);
     covMatrix = new TMatrixD(NumParamFree-1,NumParamFree-1,&matrix[0][0]);
  }
//  covMatrix->Print("f=  %10.3e  ");
  std::vector<RooRealVar> parLis;
  RooArgList *coefLis = new RooArgList();
  for (int i=0;i<numParameters;++i){
   sprintf(varName, "p%02d_%d", i,RunEra);
   parLis.emplace_back(varName,varName, coeffPoly[i].getValue(),ParMin,ParMax);
  }	      
  for (int i=0;i<numParameters;++i){
   coefLis->add(parLis[i]);
  }	      
//  gROOT->ProcessLine(".L RooBernsteinSideband.cxx+");
  BernSideBand    = new RooBernsteinSideband(Form("BernSideBand_bin%d_%d",Q2Bin,RunEra),Form("BernSideBand_bin%d_%d",Q2Bin,RunEra),*ctL,*ctK,*phi,*coefLis,maxDegree1,maxDegree2,maxDegree3);

  RooWorkspace wsb("wsb","workspace sideband");
//set to the fit range!!!
  RooRealVar *max_sbl=new RooRealVar(Form("max_sbl_bin%d_%d",Q2Bin,RunEra),Form("max_sbl_bin%d_%d",Q2Bin,RunEra),XMaxSBL);
  RooRealVar *min_sbr=new RooRealVar(Form("min_sbr_bin%d_%d",Q2Bin,RunEra),Form("min_sbr_bin%d_%d",Q2Bin,RunEra),XMinSBR);
  wsb.import(*covMatrix,Form("covMatrix_bin%d_%d",Q2Bin,RunEra));
  wsb.import(*BernSideBand);
  wsb.import(*bkg_mass_sb);
  wsb.import(*max_sbl);
  wsb.import(*min_sbr);
  wsb.writeToFile(OutSaveFileName);
  cout<<"save  workspace ==> wsb in "<<OutSaveFileName<<"\n"<<endl;
/*  double arglist[2]; 
  int err = 0;
   arglist[0]= 150000; // maximum iterations
  arglist[1]= 1.0; 
  for(int j=0;j<2;++j){
     std::vector<Variable> var; 
     double tmp_value, tmp_error;
     for(Variable &var : Minuit->getVaraibles()) {
     int index = var.getFitterIndex();
      Minuit->GetParameter(index, tmp_value, tmp_error);
      sprintf(varName, "p%d", index);
         if (index>5){
          if (tmp_value<=tmp_error && fabs(tmp_value)<0.001){
           Minuit->mnparm(index, varName, 0.,0.001, -500,500.,err);
           Minuit->FixParameter(index);
          }
          else{
           Minuit->mnparm(index, varName, fabs(tmp_value), 0.00001, 0.0,3000.,err);
          }
         }
//	std::cout<<"p("<<Index<<") = "<<" = "<<tmp_value<<std::endl ;
     }
       std::cout<<"============================================================\n"<<std::endl ;
       Minuit->mnexcm("SHOW PAR",arglist,1,err);
//       Minuit->mnexcm("MIGRAD",arglist,2,err);
         fitter.fit();
//       Minuit->Migrad();
    }   
  cout<<"                  ===*** End  Fit ***=== "<<endl;
  cout<<"                  ===*** End  Fit ***=== "<<endl;
  cout<<"                  ===*** End  Fit ***=== "<<endl;
  for(int i=0;i<numParameters;++i) {
    sprintf(varName, "p%d", i);
    double tmp_value = coeffPoly[i].getValue();
    double tmp_error = coeffPoly[i].getError();
    std::cout<<i<<" =>  "<<varName<<" = "<<tmp_value<<"+/-"<<tmp_error<<"\n"<<std::endl ;
  }
 */// 
 
  xCoeffNorm =0.0;
  xCoeffIndex=-1;
  double coeffy =0.0;
  double errory =0.0;
  std::fstream *parListOutput =  new std::fstream(ListParName,ios::out);
  std::fstream *parPlotOutput =  new std::fstream(ListPloName,ios::out);
  if(parListOutput->is_open() && parPlotOutput->is_open() ){
   std::cout<<"Open: "<<ListParName<<std::endl ;
   for(int i=0;i<numParameters;++i) {
    if( fabs(coeffPoly[i].getValue())>1. && fabs(coeffPoly[i].getError()>0.) ){
     double xCoeffNormTmp = coeffPoly[i].getValue()/fabs(coeffPoly[i].getError());
     if (xCoeffNormTmp>xCoeffNorm) {
      xCoeffNorm=xCoeffNormTmp;
      xCoeffIndex=i;
     }
    }
    if(fabs(coeffPoly[i].getValue())>fabs(coeffPoly[i].getError())){
     coeffy=  coeffPoly[i].getValue();
     errory=  coeffPoly[i].getError();
     std::cout<<Form("RESULTS==>  p(%d)=%f+/-%f rate=%f => [%s] ",i,coeffy,errory,coeffy/errory,ParCheck[i])<<std::endl;
    }else{
     coeffy=  0.0;
     errory=  0.0;
    } 
    
    *parListOutput <<std::scientific << std::setprecision(20)<< coeffy<<"+/-"<<errory<<std::endl;
    *parPlotOutput <<std::scientific << std::setprecision(20)<< coeffPoly[i].getValue()<<"+/-"<<coeffPoly[i].getError()<<std::endl;
   }
   
//
   parListOutput->close();
   parPlotOutput->close();
   std::cout<<"Close: "<<ListParName<<std::endl ;
   std::cout<<"Close: "<<ListPloName<<std::endl ;
  }else{
   if(!parListOutput->is_open()) std::cout<<"Error: can not open "<<ListParName<<std::endl ;
   if(!parPlotOutput->is_open()) std::cout<<"Error: can not open "<<ListPloName<<std::endl ;
   std::cout<<Form("Error!!!")<<std::endl ;
   exit(1);
  }
  if(xCoeffIndex>0){
   std::cout<<Form("WARNING: Normalization could be fixed better if p(%d)=1",xCoeffIndex)<<std::endl ;
  }
 
  OutFile->cd();
    
  
  TH3D* HSBFunc 	  = new TH3D( "HSBFunc"          , "HSBFunc",		NFact*xCosLHBin, XMinCosThetaL, XMaxCosThetaL,
 										NFact*xCosKHBin, XMinCosThetaK, XMaxCosThetaK,
										NFact*xPhiHBin , XMinPhi, XMaxPhi );

//   TH2D* HSBFuncXY 	  = new TH2D( "HSBFuncXY"        , "HSBFuncXY",		NFact*xCosLHBin, XMinCosThetaL, XMaxCosThetaL,
//  										NFact*xCosKHBin, XMinCosThetaK, XMaxCosThetaK);
// 
// //   TH2D* HSBFuncZY 	  = new TH2D( "HSBFuncZY"        , "HSBFuncZY",		NFact*xPhiHBin , XMinPhi, XMaxPhi,
// //  										NFact*xCosKHBin, XMinCosThetaK, XMaxCosThetaK );
// // 	
//   TH2D* HSBFuncZY 	  = new TH2D( "HSBFuncZY"        , "HSBFuncZY",		NFact*xCosKHBin, XMinCosThetaK, XMaxCosThetaK,
//  										NFact*xPhiHBin , XMinPhi, XMaxPhi);
// 	
//   TH2D* HSBFuncZX 	  = new TH2D( "HSBFuncZX"        , "HSBFuncZX",		NFact*xPhiHBin , XMinPhi, XMaxPhi,
//  										NFact*xCosLHBin, XMinCosThetaL, XMaxCosThetaL );
// 	
	
//   TH3D* HSideBandRecoTest = new TH3D( "HSideBandRecoTest", "HSideBandRecoTest", NFact*xCosLHBin, XMinCosThetaL, XMaxCosThetaL,
//  										NFact*xCosKHBin, XMinCosThetaK, XMaxCosThetaK,
// 										NFact*xPhiHBin , XMinPhi, XMaxPhi );
//   totalParams=0;
  GooFit::Observable xReco_w("xReco_w"  ,0., 2000000.)  ;
  GooFit::Observable xGene_w("xGene_w"  ,0., 2000000.)  ;
  GooFit::Observable BWidthX("BWidthX"	,0., 2.*XMaxCosThetaL);
  GooFit::Observable BWidthY("BWidthY"	,0., 2.*XMaxCosThetaK);
  GooFit::Observable BWidthZ("BWidthZ"	,0., 2.*XMaxPhi      );
  vector<GooFit::Observable> obsPolyTest;
  obsPolyTest.push_back(xCosL_x);
  obsPolyTest.push_back(xCosK_y);
  obsPolyTest.push_back(xPhiK_z);
  obsPolyTest.push_back(BWidthX);
  obsPolyTest.push_back(BWidthY);
  obsPolyTest.push_back(BWidthZ);
  vector<GooFit::Observable> dataTest;
  dataTest.push_back(xCosL_x);
  dataTest.push_back(xCosK_y);
  dataTest.push_back(xPhiK_z);
  dataTest.push_back(BWidthX);
  dataTest.push_back(BWidthY);
  dataTest.push_back(BWidthZ);
  UnbinnedDataSet* dataPlotTest = new GooFit::UnbinnedDataSet(dataTest);

  std::vector<double> DataBinContent;

  double xBinwPlot = HSBFunc->GetXaxis()->GetBinWidth(1) ;
  double yBinwPlot = HSBFunc->GetYaxis()->GetBinWidth(1) ;
  double zBinwPlot = HSBFunc->GetZaxis()->GetBinWidth(1) ;
  for(int i = 0; i < xCosLHBin*NFact ; ++i) {
   double xi = XMinCosThetaL + xBinwPlot/2.+ i*xBinwPlot;
   for(int j = 0 ; j < xCosKHBin*NFact ; ++j) {
    double yj = XMinCosThetaK + yBinwPlot/2.+ j*yBinwPlot;
    for(int k = 0 ; k < xPhiHBin*NFact  ; ++k) {
    double zk = XMinPhi + zBinwPlot/2.+ k*zBinwPlot;
//    if (xi<=XMaxCosThetaL && yj<=XMaxCosThetaK && zk<=XMaxPhi){
    xCosL_x.setValue(xi);
    xCosK_y.setValue(yj);
    xPhiK_z.setValue(zk);
   // cout <<xGene_w.getValue() <<endl;
//    xReco_w.setValue(1.);
//    xGene_w.setValue(2.);
    BWidthX.setValue(xBinwPlot);
    BWidthY.setValue(yBinwPlot);
    BWidthZ.setValue(zBinwPlot);
    
    dataPlotTest->addEvent();
    DataBinContent.push_back(HxReco->GetBinContent(i,j,k));
//    }else{
//    printf("(x,y,z)=%f %f %f (xmax,ymax,zmax) = %f %f %f\n",xi,yj,zk,XMaxCosThetaL,XMaxCosThetaK,XMaxPhi);
//    printf("(x,y,z)=%f %f %f\n",xCosL_x.getValue(),xCosK_y.getValue(),xPhiK_z.getValue());
//    }
    }
   }
  }

//      GooFit::Variable xBinWidthPlot("xBinWidth",xBinwPlot );
//      GooFit::Variable yBinWidthPlot("yBinWidth",yBinwPlot );
//      GooFit::Variable zBinWidthPlot("zBinWidth",zBinwPlot );
//      vector<GooFit::Variable> BinwsPlot;
//      BinwsPlot.push_back(xBinWidth);
//      BinwsPlot.push_back(yBinWidth);
//      BinwsPlot.push_back(zBinWidth);

  // totalParams=0;
//   GooFit::BernsteinPdf    *modelPlot=0;
//     GooFit::FastBernsteinPdf    *modelPlot=0;
//   if(integral){
//      GooFit::BernsteinTestPdf   *modelPlot     =  new GooFit::BernsteinTestPdf("modelPlot",obsPolyTest,coeffPoly,limits,BinwsPlot,maxDegree1,maxDegree2,maxDegree3);
  //}else{
//   GooFit::FastBernsteinPdf    *   modelPlot     =  new GooFit::FastBernsteinPdf("modelPlot",obsPoly,coeffPoly,limits,maxDegree1,maxDegree2,maxDegree3);
      GooFit::BernsteinTestPdf   *modelPlot     =  new GooFit::BernsteinTestPdf("modelPlot",obsPolyTest,coeffPoly,limits,maxDegree1,maxDegree2,maxDegree3,1);
//      GooFit::BernsteinTestPdf   *modelPlot     =  new GooFit::BernsteinTestPdf("modelPlot",obsPolyTest,coeffPoly,limits,maxDegree1,maxDegree2,maxDegree3,1);
  //} 


//  GooFit::BernsteinPdf    modelPlot("model",obsPolyPlot,coeffPoly,limits,Binws,maxDegree1,maxDegree2,maxDegree3);
//  GooFit::FastBernsteinPdf    modelPlot("modelPlot",obsPolyPlot,coeffPoly,limits,maxDegree1,maxDegree2,maxDegree3);
  
//  modelPlot.setData(dataReco);

//  double Vol3D = xBinwPlot*yBinwPlot*zBinwPlot;
  modelPlot->setData(dataPlotTest);
  vector<vector<double> > pdfVals_Model3D = modelPlot->getCompProbsAtDataPoints();
  double pdfVal = 0.0;
  double totalPdf = 0;
  double dataPlotEntries=dataPlotTest->getNumEvents();
  for (int i = 0; i < dataPlotEntries; ++i) {
    dataPlotTest->loadEvent(i);
    pdfVal = pdfVals_Model3D[0][i];
    totalPdf += pdfVal;
  }
  double Chi2 =0.;
  int iskip =0;
  int NDegreeofFreedomN =0;
//  int NDegreeofFreedom =-(NumParamFree+1);
  double SBEntries= HxReco->GetEntries();
//  cout <<"totalPdf = "<<totalPdf*Vol3D<<endl;
  cout <<"totalPdf = "<<totalPdf<<endl;
  cout <<"SBEntries= "<<SBEntries<<endl;
  cout <<"SBEntries= "<<dataReco->getNumEvents()<<"  (Check)"<<endl;
  cout <<"nentgrid = "<<dataPlotTest->getNumEvents()<<endl;
  double LLChi2 =0.;
  double ProbFuncTot =0;
//  int icount=0;
   for (int i = 0; i < dataPlotEntries; ++i) {
//  for (int i = 0; i < dataReco->getNumEvents(); ++i) {
    dataPlotTest->loadEvent(i);
    pdfVal = pdfVals_Model3D[0][i];
    if (pdfVal<0.0) std::cout<<"Warning!!!: SB Model "<<pdfVal <<"0 in CosL="<<xCosL_x.getValue()<<" CosK="<<xCosK_y.getValue()<<" Phi="<<xPhiK_z.getValue()<<std::endl;
//
//    HSBFunc->Fill(xCosL_x.getValue(),xCosK_y.getValue(),xPhiK_z.getValue(), SBEntries*pdfVal/totalPdf);
//
//---> perche' totalPdf e' 1...
//
    HSBFunc->Fill(xCosL_x.getValue(),xCosK_y.getValue(),xPhiK_z.getValue(), SBEntries*pdfVal);
//
//     HSBFuncXY->Fill(xCosL_x.getValue(),xCosK_y.getValue(), SBEntries*pdfVal/totalPdf);
//     HSBFuncZY->Fill(xPhiK_z.getValue(),xCosK_y.getValue(),xPhiK_z.getValue(), SBEntries*pdfVal/totalPdf);
// //   HSBFuncZY->Fill(xPhiK_z.getValue(),xCosK_y.getValue(), SBEntries*pdfVal/totalPdf);
//     HSBFuncZX->Fill(xPhiK_z.getValue(),xCosL_x.getValue(), SBEntries*pdfVal/totalPdf);
     if(pdfVal>0.) {
      double ProbFunc = SBEntries*pdfVal/totalPdf;
      double Chi2Temp = ( DataBinContent[i] - ProbFunc)*(DataBinContent[i] - ProbFunc)/ProbFunc;
      Chi2 = Chi2+( DataBinContent[i] - ProbFunc)*(DataBinContent[i] - ProbFunc)/ProbFunc;
      if(DataBinContent[i]!=0){
        LLChi2 = LLChi2+DataBinContent[i]*(log(DataBinContent[i])-log(ProbFunc));
      } else {
        iskip++;
      }	
//      if(DataBinContent[i]!=0) LLChi2 = LLChi2+DataBinContent[i]*(log(DataBinContent[i]/ProbFunc))+ProbFunc-DataBinContent[i];
      ProbFuncTot = ProbFunc+ProbFuncTot;
//       if(Chi2Temp>1.) {
//        icount++;
//        std::cout<<"======== "<<icount<<" ======================================================================\n"<<std::endl;
//        std::cout<<"==> Chi2Temp = "<<Chi2Temp<<" CosL="<<xCosL_x.getValue()<<" CosK"<<xCosK_y.getValue()<<" Phi="<<xPhiK_z.getValue()<<"\n"<<std::endl;
//        std::cout<<"==> ProbFunc ="<<ProbFunc<<" DataCont="<<DataBinContent[i]<<"\n"<<std::endl;
//       } 
      NDegreeofFreedomN++; 
     }
     //cout <<xGene_w.getValue() <<endl;
  }
  LLChi2=2*LLChi2;
  int NDegreeofFreedomR = NDegreeofFreedomN-NumParamFree;
  cout <<"Integrated Probability Function "<<ProbFuncTot <<endl;
  cout <<"In LLchi2 skipped cell = "<<iskip<<endl;
  double  PValueModelMin = TMath::Prob(Chi2, NDegreeofFreedomR);
  double  PValueModelMax = TMath::Prob(Chi2, NDegreeofFreedomN);
  double  PValueLLModelMin  = TMath::Prob(LLChi2, NDegreeofFreedomR);
  double  PValueLLModelMax  = TMath::Prob(LLChi2, NDegreeofFreedomN);
  cout <<"------------------------------------------------------- "<<endl;
  cout <<"Even Binning ("<<xCosLHBin*xCosKHBin*xPhiHBin <<" Bins) "<<endl;
  cout <<"------------------------------------------------------- "<<endl;
  cout <<"Chi2 Sideband			= "<<Chi2<<endl;
  cout <<"Chi2/NDOF			= "<<Chi2/NDegreeofFreedomN<<endl;
  cout <<"Chi2/NDOF_Reduced		= "<<Chi2/NDegreeofFreedomR<<endl;
  cout <<"P-Value		Min	= "<<PValueModelMin<<endl;
  cout <<"P-Value		Max	= "<<PValueModelMax<<endl;
  cout <<"LLChi2 Sideband		= "<<LLChi2<<endl;
  cout <<"LLChi2/NDOF			= "<<LLChi2/(NDegreeofFreedomN)<<endl;
  cout <<"LLChi2/NDOF_Reduced		= "<<LLChi2/(NDegreeofFreedomR)<<endl;
  cout <<"LL P-Value		Min	= "<<PValueLLModelMin<<endl;
  cout <<"LL P-Value		Max	= "<<PValueLLModelMax<<endl;
  cout <<"NDOF	(reduced)		= "<<NDegreeofFreedomR<<endl;
  cout <<"NDOF				= "<<NDegreeofFreedomN<<endl;
  cout <<"Num Free Param.		= "<<NumParamFree<<endl;
  cout <<"------------------------------------------------------- "<<endl;
  printf(Form("Q2Bin, Poly Degree(1,2,3),initial and final free params ==> %d & %d & %d & %d & %d & %d\\\\ \n",Q2Bin,maxDegree1,maxDegree2,maxDegree3,numParameters,NumParamFree));
  HSBFunc->Sumw2();

//======================================================================================
// Adaptive Binning GOF...
//======================================================================================  
TCanvas* ca = new TCanvas("ca","Adaptive Binning Histograms",200,200,800,800);
  ca->Divide(2,2);   
  
//
  int iCorreTagOrig = CorreAdaptX.size();
//   if( fmod(iCorreTag,MinContAdaptBin)==0){
  xAdaptNumBinC = int(iCorreTagOrig/MinContAdaptBin);
//   }else{
//    xAdaptNumBinC = int(iCorreTag/MinContAdaptBin)-1;
//   } 
//   std::cout<<Form("It will be added an extra bin with N. of entries = %d ?",iCorreTag-xAdaptNumBinC*MinContAdaptBin)<<std::endl;
  int iCorreTag=xAdaptNumBinC*MinContAdaptBin;
  std::cout<<"TKDTreeBinning Start   "<<std::endl;
  std::cout<<"TKDTreeBinning set iCorreTag	  = "<<iCorreTag<<std::endl;
  std::cout<<"TKDTreeBinning xAdaptNumBinC    = "<<xAdaptNumBinC<<std::endl;
  double *RecoAdaptC = new double[NDim*iCorreTag];
  for (int iC=0;iC<iCorreTag;iC++) { 
   RecoAdaptC[iC]	     =  CorreAdaptX[iC];
   RecoAdaptC[iC+  iCorreTag]=  CorreAdaptY[iC];
   RecoAdaptC[iC+2*iCorreTag]=  CorreAdaptZ[iC];
  } 
  RecoAdaptBinsC = new TKDTreeBinning(iCorreTag, NDim, RecoAdaptC, xAdaptNumBinC);
  int nbinsC =RecoAdaptBinsC->GetNBins();
  std::cout<<"TKDTreeBinning nbinsC    = "<<nbinsC<<std::endl;
  TH2Poly* h2polxyContC = new TH2Poly("h2polxyContC", "adapt. binning contents [CosL CosK]", RecoAdaptBinsC->GetDataMin(0), RecoAdaptBinsC->GetDataMax(0), RecoAdaptBinsC->GetDataMin(1), RecoAdaptBinsC->GetDataMax(1));
  TH2Poly* h2polxzContC = new TH2Poly("h2polxzContC", "adapt. binning contents [CosL Phi ]", RecoAdaptBinsC->GetDataMin(0), RecoAdaptBinsC->GetDataMax(0), RecoAdaptBinsC->GetDataMin(2), RecoAdaptBinsC->GetDataMax(2));
  TH2Poly* h2polxyDensC = new TH2Poly("h2polxyDensC", "adapt. binning density [CosL CosK]", RecoAdaptBinsC->GetDataMin(0), RecoAdaptBinsC->GetDataMax(0), RecoAdaptBinsC->GetDataMin(1), RecoAdaptBinsC->GetDataMax(1));
  TH2Poly* h2polxzDensC = new TH2Poly("h2polxzDensC", "adapt. binning density [CosL Phi ]", RecoAdaptBinsC->GetDataMin(0), RecoAdaptBinsC->GetDataMax(0), RecoAdaptBinsC->GetDataMin(2), RecoAdaptBinsC->GetDataMax(2));
  const double* binsMinEdgesC = RecoAdaptBinsC->GetBinsMinEdges();
  const double* binsMaxEdgesC = RecoAdaptBinsC->GetBinsMaxEdges();
  int edgeDim=0;       
  std::vector<double> DataAdaptBinContent;
  std::vector<double> Vol3DAdaptBin;
  const double* xyzvar;
  const double* xyzbinw;
//   GooFit::Observable xReco_w("xReco_w"  ,0., 2000000.)  ;
//   GooFit::Observable xGene_w("xGene_w"  ,0., 2000000.)  ;
//   GooFit::Observable BWidthX("BWidthX"	,0., 2.*XMaxCosThetaL);
//   GooFit::Observable BWidthY("BWidthY"	,0., 2.*XMaxCosThetaK);
//   GooFit::Observable BWidthZ("BWidthZ"	,0., 2.*XMaxPhi      );
  vector<GooFit::Observable> obsPolyAdapt;
  obsPolyAdapt.push_back(xCosL_x);
  obsPolyAdapt.push_back(xCosK_y);
  obsPolyAdapt.push_back(xPhiK_z);
// //   obsPolyAdapt.push_back(xReco_w);
// //   obsPolyAdapt.push_back(xGene_w);
  obsPolyAdapt.push_back(BWidthX);
  obsPolyAdapt.push_back(BWidthY);
  obsPolyAdapt.push_back(BWidthZ);
  UnbinnedDataSet* dataAdapt = new GooFit::UnbinnedDataSet(obsPolyAdapt);
  std::vector<int > AdaptExcludedEvents;
  for (int i = iCorreTag; i < iCorreTagOrig; ++i) {
  
   double point[3] = {CorreAdaptX[i],CorreAdaptY[i],CorreAdaptZ[i]};
   
   AdaptExcludedEvents.push_back(RecoAdaptBinsC->FindBin(point));
   cout<<Form("CosL=%f CosK=%f Phi=%f point = %d",CorreAdaptX[i],CorreAdaptY[i],CorreAdaptZ[i],RecoAdaptBinsC->FindBin(point))<<endl;
   
  }
  double vol =0;
  for (int i = 0; i < nbinsC; ++i) {
     edgeDim = i * NDim;
     h2polxyContC->AddBin(binsMinEdgesC[edgeDim], binsMinEdgesC[edgeDim + 1], binsMaxEdgesC[edgeDim], binsMaxEdgesC[edgeDim + 1]);
     h2polxzContC->AddBin(binsMinEdgesC[edgeDim], binsMinEdgesC[edgeDim + 2], binsMaxEdgesC[edgeDim], binsMaxEdgesC[edgeDim + 2]);
     h2polxyDensC->AddBin(binsMinEdgesC[edgeDim], binsMinEdgesC[edgeDim + 1], binsMaxEdgesC[edgeDim], binsMaxEdgesC[edgeDim + 1]);
     h2polxzDensC->AddBin(binsMinEdgesC[edgeDim], binsMinEdgesC[edgeDim + 2], binsMaxEdgesC[edgeDim], binsMaxEdgesC[edgeDim + 2]);
     xyzvar  = RecoAdaptBinsC->GetBinCenter(i);
     xyzbinw = RecoAdaptBinsC->GetBinWidth(i);
     xCosL_x.setValue(xyzvar[0]);
     xCosK_y.setValue(xyzvar[1]);
     xPhiK_z.setValue(xyzvar[2]);
//      xReco_w.setValue(1.);
//      xGene_w.setValue(2.);
     BWidthX.setValue(xyzbinw[0]);
     BWidthY.setValue(xyzbinw[1]);
     BWidthZ.setValue(xyzbinw[2]);
     vol += xyzbinw[0]*xyzbinw[1]*xyzbinw[2];
//      cout<<"================================"<<endl;
//      cout<<"NBIN    = "<<i<<endl;
//      cout<<"BWidthX = "<<xyzbinw[0]<<endl;
//      cout<<"BWidthY = "<<xyzbinw[1]<<endl;
//      cout<<"BWidthZ = "<<xyzbinw[2]<<endl;
//      cout<<"BWidthX = "<<binsMaxEdgesC[edgeDim]-binsMinEdgesC[edgeDim]<<endl;
//      cout<<"BWidthY = "<<binsMaxEdgesC[edgeDim + 1]-binsMinEdgesC[edgeDim + 1]<<endl;
//      cout<<"BWidthZ = "<<binsMaxEdgesC[edgeDim + 2]-binsMinEdgesC[edgeDim + 2]<<endl;
     dataAdapt->addEvent();
     DataAdaptBinContent.push_back(RecoAdaptBinsC->GetBinContent(i));
     Vol3DAdaptBin.push_back(xyzbinw[0]*xyzbinw[1]*xyzbinw[2]);
  }
  cout <<"Vol tot  [Adapt] = "<<vol<<endl;
  
//  
//  GooFit::FastBernsteinPdf   *modelAdapt     =  new GooFit::FastBernsteinPdf("modelAdapt",obsPolyAdapt,coeffPoly,limits,maxDegree1,maxDegree2,maxDegree3);
  GooFit::BernsteinTestPdf   *modelAdapt     =  new GooFit::BernsteinTestPdf("modelAdapt",obsPolyAdapt,coeffPoly,limits,maxDegree1,maxDegree2,maxDegree3,1);
//

  modelAdapt->setData(dataAdapt);
  vector<vector<double> > pdfVals_Model3DAdapt = modelAdapt->getCompProbsAtDataPoints();
  pdfVal = 0.0;
  totalPdf = 0;
  double dataAdaptEntries=dataAdapt->getNumEvents();
  for (int i = 0; i < dataAdaptEntries; ++i) {
    dataAdapt->loadEvent(i);
    pdfVal = pdfVals_Model3DAdapt[0][i];
    totalPdf += pdfVals_Model3DAdapt[0][i];
//    totalPdf += pdfVals_Model3DAdapt[0][i]*Vol3DAdaptBin[i];
  }
  cout <<"totalPdf [Adapt] = "<<totalPdf<<endl;
//  cout <<Form("dataAdaptEntries [%f]must be = nbinsC [%d] ",dataAdaptEntries,nbinsC)<<endl;
  double AdaptProbFuncTot =0;
//double   icount = 0;
  double lambda = 2./3.;
  double AdaptChi2 = 0;
  double AdaptLLChi2 = 0;
  double AdaptPowerDivergence = 0. ;
  
  int NDegreeofFreedomAdaptN = 0;  
//  int NDegreeofFreedomAdapt =-(NumParamFree);  
  for (int ii = 0; ii < nbinsC; ++ii){
       h2polxyContC->SetBinContent(ii+1, RecoAdaptBinsC->GetBinContent(ii));
       h2polxzContC->SetBinContent(ii+1, RecoAdaptBinsC->GetBinContent(ii));
       h2polxyDensC->SetBinContent(ii+1, RecoAdaptBinsC->GetBinDensity(ii));
       h2polxzDensC->SetBinContent(ii+1, RecoAdaptBinsC->GetBinDensity(ii));
       dataAdapt->loadEvent(ii);
       pdfVal = pdfVals_Model3DAdapt[0][ii];
       for (int iii = 0; iii < iCorreTagOrig-iCorreTag; ++iii) {
        if(AdaptExcludedEvents[iii]==ii){
	 DataAdaptBinContent[ii]++;
        }
       }	 	

       if( DataAdaptBinContent[ii]!=MinContAdaptBin) std::cout<<Form("DataAdaptBinContent[%d]=%f",ii,DataAdaptBinContent[ii])<<std::endl;
//       if(pdfVal!=0.&& DataAdaptBinContent[ii]==MinContAdaptBin) {
       if(pdfVal!=0.) {
//        double ProbFunc = pdfVal/totalPdf;
//        double ProbFunc = iCorreTag*pdfVal*Vol3DAdaptBin[ii]/totalPdf;
        double ProbFunc = SBEntries*pdfVal/totalPdf;
	AdaptPowerDivergence = AdaptPowerDivergence+ (DataAdaptBinContent[ii])*(pow(DataAdaptBinContent[ii]/ProbFunc,lambda)-1);

      double AdaptChi2Temp = ( DataAdaptBinContent[ii] - ProbFunc)*(DataAdaptBinContent[ii] - ProbFunc)/(ProbFunc);
//      AdaptLLChi2 = AdaptLLChi2+DataAdaptBinContent[ii]*(log(DataAdaptBinContent[ii])-log(ProbFunc));
      AdaptLLChi2 = AdaptLLChi2+DataAdaptBinContent[ii]*(log(DataAdaptBinContent[ii])-log(ProbFunc))+ProbFunc-DataAdaptBinContent[ii];
//      AdaptChi2 = AdaptChi2+ AdaptChi2Temp;
      AdaptChi2 = AdaptChi2+ ( DataAdaptBinContent[ii] - ProbFunc)*(DataAdaptBinContent[ii] - ProbFunc)/(ProbFunc);
      AdaptProbFuncTot = ProbFunc+AdaptProbFuncTot;
//        if(AdaptChi2Temp>1.) {
// 	icount++;
// 	std::cout<<"======== "<<icount<<" ======================================================================\n"<<std::endl;
// 	std::cout<<"==> AdaptChi2Temp = "<<AdaptChi2Temp<<" CosL="<<xCosL_x.getValue()<<" CosK"<<xCosK_y.getValue()<<" Phi="<<xPhiK_z.getValue()<<"\n"<<std::endl;
// 	std::cout<<"==> ProbFunc ="<<ProbFunc<<" DataAdaptCont="<<DataAdaptBinContent[ii]<<"\n"<<std::endl;
//        } 
      NDegreeofFreedomAdaptN++;
     }
   }
  AdaptPowerDivergence = 2.*AdaptPowerDivergence/(lambda+1)/lambda ;	
  cout <<"Integrated Probability Function [Adapt]"<<ProbFuncTot <<endl;
  int NDegreeofFreedomAdaptR = NDegreeofFreedomAdaptN-NumParamFree;
  AdaptLLChi2=2*AdaptLLChi2;
  PValueModelMin = TMath::Prob(AdaptChi2, NDegreeofFreedomAdaptR);
  PValueModelMax = TMath::Prob(AdaptChi2, NDegreeofFreedomAdaptN);
  PValueLLModelMin =  TMath::Prob(AdaptLLChi2, NDegreeofFreedomAdaptR);
  PValueLLModelMax =  TMath::Prob(AdaptLLChi2, NDegreeofFreedomAdaptN);
  double PValuePDModelMin =  TMath::Prob(AdaptPowerDivergence, NDegreeofFreedomAdaptR);
  double PValuePDModelMax =  TMath::Prob(AdaptPowerDivergence, NDegreeofFreedomAdaptN);
  cout <<"------------------------------------------------------- "<<endl;
  cout <<"Adaptive Binning ("<<xAdaptNumBinC <<" Bins)\n"          <<endl;
  cout <<"MinContAdaptBin = "<<MinContAdaptBin                     <<endl;
  cout <<"------------------------------------------------------- "<<endl;
  cout <<"Chi2 Sideband			[Adapt] = "<<AdaptChi2<<endl;
  cout <<"Chi2/NDOF			[Adapt] = "<<AdaptChi2/NDegreeofFreedomAdaptR<<endl;
  cout <<"Chi2/NDOF_Reduced		[Adapt] = "<<AdaptChi2/NDegreeofFreedomAdaptN<<endl;
  cout <<"P-Value			Min	[Adapt] = "<<PValueModelMin<<endl;
  cout <<"P-Value			Max	[Adapt] = "<<PValueModelMax<<endl;
  cout <<"LL Chi2 Sideband		[Adapt] = "<<AdaptLLChi2<<endl;
  cout <<"LL Chi2/NDOF			[Adapt] = "<<AdaptLLChi2/NDegreeofFreedomAdaptN<<endl;
  cout <<"LL Chi2/NDOF_Reduced		[Adapt] = "<<AdaptLLChi2/NDegreeofFreedomAdaptR<<endl;
  cout <<"LL P-Value		Min	[Adapt] = "<<PValueLLModelMin<<endl;
  cout <<"LL P-Value		Max	[Adapt] = "<<PValueLLModelMax<<endl;
  cout <<"PowerDivergence			[Adapt] = "<<AdaptPowerDivergence<<endl;
  cout <<"PowerDivergence/NDOF		[Adapt] = "<<AdaptPowerDivergence/NDegreeofFreedomAdaptN<<endl;
  cout <<"PowerDivergence/NDOF_Reduced	[Adapt] = "<<AdaptPowerDivergence/NDegreeofFreedomAdaptR<<endl;
  cout <<"PD P-Value		Min	[Adapt] = "<<PValuePDModelMin<<endl;
  cout <<"PD P-Value		Max	[Adapt] = "<<PValuePDModelMax<<endl;
  cout <<"NDOF	(reduced)		[Adapt] = "<<NDegreeofFreedomAdaptR<<endl;
  cout <<"NDOF				[Adapt] = "<<NDegreeofFreedomAdaptN<<endl;
  cout <<"Num Free Param.	   	   = "<<NumParamFree<<endl;
  cout <<"------------------------------------------------------- "<<endl;

  ca->cd(1);
//  h2polxyContC->Draw("lego");
  h2polxyContC->Draw("COLZ L");
  ca->Update();   
  ca->cd(2);
//  h2polxzContC->Draw("lego");
  h2polxzContC->Draw("COLZ L");
  ca->Update();   
  ca->cd(3);
  h2polxyDensC->Draw("COLZ L");
  ca->Update();   
  ca->cd(4);
  h2polxzDensC->Draw("COLZ L");
  ca->Update();   
 //    h2polxyC->Draw("LEGO");
//    ca->Update();   
//    ca->cd(4);
//    h2polxzC->Draw("LEGO");
//    ca->Update();   
//
// Adaptive Binning...End
//  



//  cout<<std::scientific << std::setprecision(40)<<"Norm PDF SideBand = "<<modelPlot->normalize()<<";"<<endl;
//==== closure test plots
//   modelPlot->setData(dataPlot);
//   vector<vector<double> > pdfVals_Model3D_Reco = modelPlot->getCompProbsAtDataPoints();
//   pdfVal = 0.0;
//   for (int i = 0; i < dataPlot->getNumEvents(); ++i) {
//     dataReco->loadEvent(i);
//     pdfVal = pdfVals_Model3D_Reco[0][i];
//     if (pdfVal<0.0) std::cout<<"Warning!!!: Effi Model Reco Test"<<pdfVal <<"0 in CosL="<<xCosL_x.getValue()<<" CosK="<<xCosK_y.getValue()<<" Phi="<<xPhiK_z.getValue()<<std::endl;
// //     double xL = xCosL_x.getValue();
// //     double yK = xCosK_y.getValue();
// //     double zP = xPhiK_z.getValue();
// //    HSBFunc->Fill(xL,yK,zP, pdfVal);
//      HSideBandRecoTest->Fill(xCosL_x.getValue(),xCosK_y.getValue(),xPhiK_z.getValue(), pdfVal*xGene_w.getValue());
// //     HSideBandCosLFunc->Fill(xL, pdfVal);
// //     HSideBandCosKFunc->Fill(yK, pdfVal);
// //     HSideBandPhiFunc ->Fill(zP, pdfVal);
//   }
//  HSideBandRecoTest->Sumw2();

  TH1D* HSBFuncX  = (TH1D*) HSBFunc->ProjectionX("HSBFuncX",1,HSBFunc->GetNbinsY(),1,HSBFunc->GetNbinsZ());HSBFuncX->SetTitle(Form("Cos#theta_{L} Projection [q^{2} bin %d run %d]",Q2Bin,RunEra));
  TH1D* HSBFuncY  = (TH1D*) HSBFunc->ProjectionY("HSBFuncY",1,HSBFunc->GetNbinsX(),1,HSBFunc->GetNbinsZ());HSBFuncY->SetTitle(Form("Cos#theta_{K} Projection [q^{2} bin %d run %d]",Q2Bin,RunEra));
  TH1D* HSBFuncZ  = (TH1D*) HSBFunc->ProjectionZ("HSBFuncZ",1,HSBFunc->GetNbinsX(),1,HSBFunc->GetNbinsY());HSBFuncZ->SetTitle(Form("#varphi Projection [q^{2} bin %d run %d]",Q2Bin,RunEra));
  TH2D* HSBFuncXY = (TH2D*) HSBFunc->Project3D("xy");HSBFuncXY->SetTitle(Form("2D Model Projection (Cos#theta_{l},Cos#theta_{k})    [q^{2} bin %d Run II %d]",Q2Bin,RunEra));
  TH2D* HSBFuncZY = (TH2D*) HSBFunc->Project3D("zy");HSBFuncZY->SetTitle(Form("2D Model Projection (#varphi,Cos#theta_{k})    [q^{2} bin %d Run II %d]",Q2Bin,RunEra));
  TH2D* HSBFuncZX = (TH2D*) HSBFunc->Project3D("zx");HSBFuncZX->SetTitle(Form("2D Model Projection (#varphi,Cos#theta_{l})   [q^{2} bin %d Run II %d]",Q2Bin,RunEra));
//  gStyle->SetOptStat(111111);
//  gStyle -> SetOptFit(111111);
  
   TH3D *HSideBand3D  = (TH3D*)HxReco->Clone(); HSideBand3D->SetName("HSideBand3D");HSideBand3D->Sumw2();
  
  
  
   TH1D* HSideBand3DX  = (TH1D*) HSideBand3D->ProjectionX("HSideBand3DX",1,HSideBand3D->GetNbinsY(),1,HSideBand3D->GetNbinsZ());HSideBand3DX->SetTitle(Form("Cos#theta_{L} Projection [q^{2} bin %d run %d]",Q2Bin,RunEra));
   TH1D* HSideBand3DY  = (TH1D*) HSideBand3D->ProjectionY("HSideBand3DY",1,HSideBand3D->GetNbinsX(),1,HSideBand3D->GetNbinsZ());HSideBand3DY->SetTitle(Form("Cos#theta_{K} Projection [q^{2} bin %d run %d]",Q2Bin,RunEra));
   TH1D* HSideBand3DZ  = (TH1D*) HSideBand3D->ProjectionZ("HSideBand3DZ",1,HSideBand3D->GetNbinsX(),1,HSideBand3D->GetNbinsY());HSideBand3DZ->SetTitle(Form("#varphi Projection [q^{2} bin %d run %d]",Q2Bin,RunEra));
   TH2D* HSideBand3DXY = (TH2D*) HSideBand3D->Project3D("xy");HSideBand3DXY->SetTitle(Form("2D Projection (Cos#theta_{l},Cos#theta_{k})    [q^{2} bin %d Run II %d]",Q2Bin,RunEra));
   TH2D* HSideBand3DZY = (TH2D*) HSideBand3D->Project3D("zy");HSideBand3DZY->SetTitle(Form("2D Projection (#varphi,Cos#theta_{k})    [q^{2} bin %d Run II %d]",Q2Bin,RunEra));
   TH2D* HSideBand3DZX = (TH2D*) HSideBand3D->Project3D("zx");HSideBand3DZX->SetTitle(Form("2D Projection (#varphi,Cos#theta_{l})    [q^{2} bin %d Run II %d]",Q2Bin,RunEra));


//   TH1D*HSideBand3DY_1=(TH1D*) HSideBand3DZY->ProjectionY("HSideBand3DY_1",1,HSideBand3DZY->GetNbinsZ());
//   TH1D*HSideBand3DZ_1=(TH1D*) HSideBand3DZY->ProjectionX("HSideBand3DZ_1",1,HSideBand3DZY->GetNbinsY());

//////////////////////////////////
//
// SideBand Plots
//
//////////////////////////////////
  gStyle->SetTitleBorderSize(0);
  gStyle->SetTitleFontSize(0.05) ;
  c6->cd(1);
//  HSideBand3DX->Draw("E1");
//  HSideBandX->Draw("E1");
  TH1D* HSBFuncX_ratio = (TH1D*)HSBFuncX->Clone();
  HSBFuncX_ratio->Rebin(NFact);
  HSBFuncX_ratio->SetLineWidth(2.);
  HSBFuncX_ratio->SetLineColor(kRed);
//  HSBFuncX->Scale(HSideBandX->Integral()/HSBFuncX->Integral()*NFact);
//  HSBFuncX->Scale(HSideBand3DX->Integral()/HSBFuncX->Integral());
//  HSBFuncX->Draw("same,HIST C");
  HSideBand3DX->SetMinimum(SetMinProj);
  RatioDataModel3DX = new TRatioPlot(HSideBand3DX,HSBFuncX_ratio);
  RatioDataModel3DX->SetGraphDrawOpt("L");
  RatioDataModel3DX->SetSeparationMargin(0.0);
  RatioDataModel3DX->SetH1DrawOpt("E1");
  RatioDataModel3DX->SetH2DrawOpt("HIST C");
  RatioDataModel3DX->Draw();
  RatioDataModel3DX->GetUpperPad()->cd();;
  HSBFuncX->Scale(NFact);
  HSBFuncX->Draw("same HIST C");
  RatioDataModel3DX->GetLowerRefGraph()->SetMinimum(SetMinRatio);
  RatioDataModel3DX->GetLowerRefGraph()->SetMaximum(SetMaxRatio);
  c6->Update();
  c6->cd(2);
//  HSideBandY->Draw("E1");
//  HSideBand3DY->Draw("E1");
  TH1D* HSBFuncY_ratio = (TH1D*)HSBFuncY->Clone();
  HSBFuncY_ratio->Rebin(NFact);
  HSBFuncY_ratio->SetLineWidth(2.);
  HSBFuncY_ratio->SetLineColor(kRed);
//  HSBFuncY->Scale(HSideBand3DY->Integral()/HSBFuncY->Integral());
//  HSBFuncY->Scale(HSideBandY->Integral()/HSBFuncY->Integral()*NFact);
//  HSBFuncY->Draw("same,HIST C");
  HSideBand3DY->SetMinimum(SetMinProj);
  RatioDataModel3DY = new TRatioPlot(HSideBand3DY,HSBFuncY_ratio);
  RatioDataModel3DY->SetGraphDrawOpt("L");
  RatioDataModel3DY->SetSeparationMargin(0.0);
  RatioDataModel3DY->SetH1DrawOpt("E1");
  RatioDataModel3DY->SetH2DrawOpt("HIST C");
  RatioDataModel3DY->Draw();
  RatioDataModel3DY->GetUpperPad()->cd();;
  HSBFuncY->Scale(NFact);
  HSBFuncY->Draw("same HIST C");
  RatioDataModel3DY->GetLowerRefGraph()->SetMinimum(SetMinRatio);
  RatioDataModel3DY->GetLowerRefGraph()->SetMaximum(SetMaxRatio);
  TLegend* leg_SBFunc = new TLegend(0.40,0.67,0.90,0.90);
  leg_SBFunc->SetTextSize(0.025) ;
  leg_SBFunc->SetTextAlign(13);
  leg_SBFunc->SetBorderSize(0.);
  leg_SBFunc->SetFillStyle(0);
//  leg_SBFunc->SetEntrySeparation(0.0001);
//  leg_SBFunc->SetNColumns(1);
//   leg_SBFunc->AddEntry(HSBFuncY ,Form( "%5.3f<#Chi^{2}_{/NDOF}<%5.3f", AdaptChi2/NDegreeofFreedomAdaptN,AdaptChi2/NDegreeofFreedomAdaptR),"");
//   leg_SBFunc->AddEntry(HSBFuncY ,Form( "reduced NDOF=%d [NDOF=%d] ", NDegreeofFreedomAdaptR,NDegreeofFreedomAdaptN),"");
//   leg_SBFunc->AddEntry(HSBFuncY ,Form( "%6.5f<P-Value Min<%6.5f ", PValueModelMin,PValueModelMax),"");
//   leg_SBFunc->AddEntry(HSBFuncY ,Form( "Events x  adapt. bin=%d ",MinContAdaptBin),"");
//   leg_SBFunc->AddEntry(HSBFuncY ,Form( "%5.3f<#Chi^{2}_{/NDOF}<%5.3f", AdaptPowerDivergence/NDegreeofFreedomAdaptN,AdaptPowerDivergence/NDegreeofFreedomAdaptR),"");
//   leg_SBFunc->AddEntry(HSBFuncY ,Form( "reduced NDOF=%d [NDOF=%d] ", NDegreeofFreedomAdaptR,NDegreeofFreedomAdaptN),"");
//   leg_SBFunc->AddEntry(HSBFuncY ,Form( "%6.5f<P-Value<%6.5f ", PValuePDModelMin,PValuePDModelMax),"");
//   leg_SBFunc->AddEntry(HSBFuncY ,Form( "num of Events x [adaptive] bin=%d ",MinContAdaptBin),"");
  leg_SBFunc->AddEntry(RatioDataModel3DY ,Form( "%5.3f<#Chi^{2}_{/NDOF}<%5.3f", AdaptPowerDivergence/NDegreeofFreedomAdaptN,AdaptPowerDivergence/NDegreeofFreedomAdaptR),"");
  leg_SBFunc->AddEntry(RatioDataModel3DY ,Form( "reduced NDOF=%d [NDOF=%d] ", NDegreeofFreedomAdaptR,NDegreeofFreedomAdaptN),"");
  leg_SBFunc->AddEntry(RatioDataModel3DY ,Form( "%6.5f<P-Value<%6.5f ", PValuePDModelMin,PValuePDModelMax),"");
  leg_SBFunc->AddEntry(RatioDataModel3DY ,Form( "num of Events x [adaptive] bin=%d ",MinContAdaptBin),"");
  leg_SBFunc->Draw();
  c6->Update();
  c6->cd(3);
//  HSideBandZ->Draw("E1");
//  HSideBand3DZ->Draw("E1");
  TH1D* HSBFuncZ_ratio = (TH1D*)HSBFuncZ->Clone();
  HSBFuncZ_ratio->Rebin(NFact);
  HSBFuncZ_ratio->SetLineWidth(2.);
  HSBFuncZ_ratio->SetLineColor(kRed);
//  HSBFuncZ->Scale(HSideBandZ->Integral()/HSBFuncZ->Integral()*NFact);
//  HSBFuncZ->Scale(HSideBand3DZ->Integral()/HSBFuncZ->Integral());
//  HSBFuncZ->Draw("same,HIST C");
  HSideBand3DZ->SetMinimum(SetMinProj);
  RatioDataModel3DZ = new TRatioPlot(HSideBand3DZ,HSBFuncZ_ratio);
  RatioDataModel3DZ->SetGraphDrawOpt("L");
  RatioDataModel3DZ->SetSeparationMargin(0.0);
  RatioDataModel3DZ->SetH1DrawOpt("E1");
  RatioDataModel3DZ->SetH2DrawOpt("HIST C");
  RatioDataModel3DZ->Draw();
  RatioDataModel3DZ->GetUpperPad()->cd();;
  HSBFuncZ->Scale(NFact);
  HSBFuncZ->Draw("same HIST C");
  RatioDataModel3DZ->GetLowerRefGraph()->SetMinimum(SetMinRatio);
  RatioDataModel3DZ->GetLowerRefGraph()->SetMaximum(SetMaxRatio);
  RatioDataModel3DZ->GetUpperRefXaxis()->SetLimits(-TMath::Pi(),TMath::Pi());
  RatioDataModel3DZ->GetLowerRefGraph()->GetXaxis()->SetLimits(-TMath::Pi(),TMath::Pi());
  c6->Update();

  c6->cd(4); 
  TH1D* pdfHxMassQ2Set = (TH1D*)pdfHxMassQ2->Clone();
  HxMassQ2   ->GetXaxis()->SetRangeUser(HistMassL1,HistMassL2);
  HxMassQ2SB ->GetXaxis()->SetRangeUser(XMinSBL,XMaxSBR);
  pdfHxMassQ2->GetXaxis()->SetRangeUser(XMinSBL,XMaxSBR);
  sigHxMassQ2->GetXaxis()->SetRangeUser(XMinSBL,XMaxSBR);
  bkgHxMassQ2->GetXaxis()->SetRangeUser(XMinSBL,XMaxSBR);
  pdfHxMassQ2Set->GetXaxis()->SetRangeUser(XLeftSet,XRightSet);
//   HxMassQ2   ->GetXaxis()->SetLimits(HistMassL1,HistMassL2);
//   HxMassQ2SB ->GetXaxis()->SetLimits(HistMassL1,HistMassL2);
//   pdfHxMassQ2->GetXaxis()->SetLimits(HistMassL1,HistMassL2);
//   sigHxMassQ2->GetXaxis()->SetLimits(HistMassL1,HistMassL2);
//   bkgHxMassQ2->GetXaxis()->SetLimits(HistMassL1,HistMassL2);
//  pdfHxMassQ2Set->GetXaxis()->SetLimits(HistMassL1,HistMassL2);
//  sigHxMassQ2->SetLineStyle(kDashed);
//  bkgHxMassQ2->SetLineStyle(kDashed);
  pdfHxMassQ2Set->SetFillColor(kBlue);
  pdfHxMassQ2Set->SetFillStyle(3013);
  pdfHxMassQ2Set->SetLineWidth(1.0);
//  
  TLegend* leg_signSB = new TLegend(0.25,0.75,0.90,0.90);
  leg_signSB->SetTextSize(0.025) ;
  leg_signSB->SetTextAlign(11);
  leg_signSB->SetBorderSize(0.);
  leg_signSB->SetFillStyle(0);
//  leg_signSB->AddEntry(HxMassQ2 ,Form( "in red:"),"");
  leg_signSB->AddEntry(HxMassQ2 ,Form( "#color[2]{sideband for q^{2} bin %d [%2.1f<q^{2}<%2.1f]}", Q2Bin, Q2Min,Q2Max),"");
  if(SigmaProbSign==0){
   leg_signSB->AddEntry(HxMassQ2 ,Form( "sb entries=%4.0f [-%s#sigma,-%s#sigma]&[%s#sigma,%s#sigma]",\\
    SBEntries,FMTNSigma1L,FMTNSigma2L,FMTNSigma1R,FMTNSigma2R),"");
   leg_signSB->AddEntry(HxMassQ2 ,Form( "bckg entries=%4.0f [-2#sigma,2#sigma]",NBckgInt2Sigma),"");
  }else{
   leg_signSB->AddEntry(HxMassQ2 ,Form( "sb entries=%4.0f",SBEntries),"");
   leg_signSB->AddEntry(HxMassQ2 ,Form( "bckg entries=%4.0f [95.5%% of signal]",NBckgInt2Sigma),"");
  }  
//  leg_signSB->AddEntry(HxMassQ2 ,Form( "bckg entries=%4.0f [-2#sigma,2#sigma]",NBckgInt2Sigma),"");
//  leg_signSB->AddEntry(HxMassQ2 ,Form( "#chi^{2}_{/NDOF}=%5.2f NDOF=%d", AdaptChi2/NDegreeofFreedomAdapt,NDegreeofFreedomAdapt),"pel");
// //  leg_signSB->SetHeader(Form( "B_{0} #rightarrow K^{*0}#mu^{+}#mu^{-} Mass Spectrum for bin %d [%2.1f<q^{2}<%2.1f] ", Q2Bin, Q2Min,Q2Max));
//   leg_signSB->AddEntry(HxMassQ2 ,Form( "B_{0} #rightarrow K^{*0}#mu^{+}#mu^{-} Mass Spectrum for bin %d [%2.1f<q^{2}<%2.1f] ", Q2Bin, Q2Min,Q2Max),"pel");
//   leg_signSB->AddEntry(HxMassQ2SB ,Form( "B_{0} #rightarrow K^{*0}#mu^{+}#mu^{-} Mass Spectrum for bin %d [%2.1f<q^{2}<%2.1f] ", Q2Bin, Q2Min,Q2Max),"pel");
//  gStyle->SetTitleFontSize(0.09) ;
  HxMassQ2->SetMaximum(1.5 * HxMassQ2->GetMaximum());
  HxMassQ2->SetTitle(Form("B_{0} #rightarrow K^{*0}#mu^{+}#mu^{-} Mass Spectrum [q^{2} bin %d run %d]",Q2Bin,RunEra));
//  HxMassQ2->SetTitle(Form( "B_{0} #rightarrow K^{*0}#mu^{+}#mu^{-} Mass Spectrum and sideband for q^{2} bin %d [%2.1f<q^{2}<%2.1f] ", Q2Bin, Q2Min,Q2Max));
  HxMassQ2->SetLineColor(kBlue);
  HxMassQ2->DrawCopy("E1,9");
  HxMassQ2->GetXaxis()->SetTitle("Mass (GeV/c^{2})");
//  HxMassQ2SB->SetTitleSize(20);
  HxMassQ2SB->SetLineColor(kOrange);
  HxMassQ2SB->SetTitle(Form( "B_{0} #rightarrow K^{*0}#mu^{+}#mu^{-} Mass Spectrum  and sideband for q^{2} bin %d [%2.1f<q^{2}<%2.1f] ", Q2Bin, Q2Min,Q2Max));
  HxMassQ2SB->SetFillColor(kRed);
  HxMassQ2SB->DrawCopy("SAME,B,9");
  pdfHxMassQ2->DrawCopy("same,HIST C,9");
  sigHxMassQ2->DrawCopy("same,HIST C,9");
  bkgHxMassQ2->DrawCopy("same,HIST C,9");
  pdfHxMassQ2Set->DrawCopy("same,HIST C,9");
  leg_signSB->Draw();
  c6->Update();
  
//////////////////////////////////
//////////////////////////////////
  gStyle->SetHistLineStyle(1);
//  
// separate plot for ProjX 
//  
  cprojX->cd();
//  HSBFuncX->Scale(NFact);
//   HSideBand3DX->GetXaxis()->SetTitle("Cos#theta_{l}");
//   HSideBand3DX->Draw("E1");
// //  HSideBandX->Draw("E1");
  HSBFuncX->SetLineColor(kRed);
  HSBFuncX->SetLineWidth(2.);
// //  HSBFuncX->Scale(HSideBand3DX->Integral()/HSBFuncX->Integral()*NFact);
// //  HSBFuncX->Scale(HSideBand3DX->Integral()/HSBFuncX->Integral());
//  HSBFuncX->Draw("same,HIST C");
  TRatioPlot* RatioProjModel3DX = new TRatioPlot(HSideBand3DX,HSBFuncX_ratio);
  RatioProjModel3DX->SetGraphDrawOpt("L");
  RatioProjModel3DX->SetSeparationMargin(0.0);
  RatioProjModel3DX->SetH1DrawOpt("E1");
  RatioProjModel3DX->SetH2DrawOpt("HIST C");
  RatioProjModel3DX->Draw();
  RatioProjModel3DX->GetUpperPad()->cd();;
  HSBFuncX->Draw("same HIST C");
  RatioProjModel3DX->GetLowerRefYaxis()->SetTitle("Ratio");
  RatioProjModel3DX->GetUpperRefYaxis()->SetTitle(Form("Events/%2.2f",(XMaxCosThetaL-XMinCosThetaL)/xCosLHBin));  
  RatioProjModel3DX->GetLowerRefGraph()->SetMinimum(SetMinRatio);
  RatioProjModel3DX->GetLowerRefGraph()->SetMaximum(SetMaxRatio);
  cprojX->Update();

//  
// separate plot for ProjY 
//  
  cprojY->cd();
// //  HSBFuncY->Scale(NFact);
// //  HSideBandY->Draw("E1");
//   HSideBand3DY->GetXaxis()->SetTitle("Cos#theta_{k}");
//   HSideBand3DY->Draw("E1");
  HSBFuncY->SetLineColor(kRed);
  HSBFuncY->SetLineWidth(2.);
//   HSBFuncY->Draw("same,HIST C");
  TRatioPlot* RatioProjModel3DY = new TRatioPlot(HSideBand3DY,HSBFuncY_ratio);
  RatioProjModel3DY->SetGraphDrawOpt("L");
  RatioProjModel3DY->SetSeparationMargin(0.0);
  RatioProjModel3DY->SetH1DrawOpt("E1");
  RatioProjModel3DY->SetH2DrawOpt("HIST C");
  RatioProjModel3DY->Draw();
  RatioProjModel3DY->GetUpperPad()->cd();;
  HSBFuncY->Draw("same HIST C");
  RatioProjModel3DY->GetLowerRefYaxis()->SetTitle("Ratio");
  RatioProjModel3DY->GetUpperRefYaxis()->SetTitle(Form("Events/%2.2f",(XMaxCosThetaK-XMinCosThetaK)/xCosKHBin));  
  RatioProjModel3DY->GetLowerRefGraph()->SetMinimum(SetMinRatio);
  RatioProjModel3DY->GetLowerRefGraph()->SetMaximum(SetMaxRatio);
  TLegend* leg_SBProj = new TLegend(0.42,0.60,0.90,0.85);
  leg_SBProj->SetTextSize(0.025) ;
  leg_SBProj->SetTextAlign(13);
  leg_SBProj->SetBorderSize(0.);
  leg_SBProj->SetFillStyle(0);
  leg_SBProj->AddEntry(HSideBand3DY ,Form( "Data Projection"),"lep");
  leg_SBProj->AddEntry(HSBFuncY_ratio ,Form( "Fit  Projection"),"l");
  leg_SBProj->AddEntry(RatioProjModel3DY ,Form( "#chi^{2}_{/NDOF} #approx %3.2f", AdaptPowerDivergence/NDegreeofFreedomAdaptR),"");
  leg_SBProj->AddEntry(RatioProjModel3DY ,Form( "p-value #approx %3.2f ", PValuePDModelMin),"");
  leg_SBProj->Draw();
//   RatioProjModel3DY->GetUpperPad()->cd();;
//   HSBFuncY->Draw("same HIST C");
  cprojY->Update();
// separate plot for ProjZ 
  cprojZ->cd();
// //  HSBFuncZ->Scale(NFact);
// //  HSideBandZ->Draw("E1");
//   HSideBand3DZ->GetXaxis()->SetTitle("#varphi");
//   HSideBand3DZ->Draw("E1");
// //  HSBFuncZ->Scale(HSideBand3DZ->Integral()/HSBFuncZ->Integral()*NFact);
// //  HSBFuncZ->Scale(HSideBand3DZ->Integral()/HSBFuncZ->Integral());

  HSBFuncZ->SetLineColor(kRed);
  HSBFuncZ->SetLineWidth(2.);
  HSBFuncZ->Draw("same,HIST C");
  HSideBand3DZ->SetMinimum(SetMinProj);
  TRatioPlot* RatioProjModel3DZ = new TRatioPlot(HSideBand3DZ,HSBFuncZ_ratio);
  RatioProjModel3DZ->SetGraphDrawOpt("L");
  RatioProjModel3DZ->SetSeparationMargin(0.0);
  RatioProjModel3DZ->SetH1DrawOpt("E1");
  RatioProjModel3DZ->SetH2DrawOpt("HIST C");
  RatioProjModel3DZ->Draw();
  RatioProjModel3DZ->GetUpperPad()->cd();
  HSBFuncZ->SetTitle("");
  HSBFuncZ->Draw("same HIST C");
  RatioProjModel3DZ->GetLowerRefYaxis()->SetTitle("Ratio");
  RatioProjModel3DZ->GetUpperRefYaxis()->SetTitle(Form("Events/%2.2f",(XMaxPhi-XMinPhi)/xPhiHBin));  
  RatioProjModel3DZ->GetLowerRefGraph()->SetMinimum(SetMinRatio);
  RatioProjModel3DZ->GetLowerRefGraph()->SetMaximum(SetMaxRatio);
  RatioProjModel3DZ->GetUpperRefXaxis()->SetLimits(-TMath::Pi(),TMath::Pi());
  RatioProjModel3DZ->GetLowerRefGraph()->GetXaxis()->SetLimits(-TMath::Pi(),TMath::Pi());
  cprojZ->Update();

  gStyle->SetPalette(57);
  cxy->cd();
  gPad->SetTheta(40.);
  gPad->SetPhi(40.);
  //HSideBand3DXY->Rebin2D(2,2);
  HSideBand3DXY->Smooth(2);
  HSideBand3DXY->SetFillColor(38);
//  HSideBand3DXY->Draw("SURF2 0");
  HSideBand3DXY->GetXaxis()->SetLabelSize(0);
  HSideBand3DXY->GetYaxis()->SetLabelSize(0);
  HSideBand3DXY->GetZaxis()->SetLabelSize(0);
  HSideBand3DXY->GetXaxis()->SetTitle("Cos#theta_{l}");
  HSideBand3DXY->GetYaxis()->SetTitle("Cos#theta_{k}");
  HSideBand3DXY->Draw("LEGO2 0 fbbb");
//  auto cutg = new TCutG("cutg",5);
//  cxy->cd(2);
//  HSBFuncXY->Scale(HSideBand3DXY->Integral()/HSBFuncXY->Integral()*NFact);
  HSBFuncXY->SetLineColor(kRed);
  HSBFuncXY->GetXaxis()->SetTitle("Cos#theta_{l}");
  HSBFuncXY->GetYaxis()->SetTitle("Cos#theta_{k}");
  HSBFuncXY->Draw("SURF A same fbbb");
  
//
  cyz->cd();
  gPad->SetTheta(40.);
  gPad->SetPhi(40.);
//  HSideBand3DZY->Rebin2D(2,2);
  HSideBand3DZY->Smooth(2);
  HSideBand3DZY->GetXaxis()->SetLabelSize(0);
  HSideBand3DZY->GetYaxis()->SetLabelSize(0);
  HSideBand3DZY->GetZaxis()->SetLabelSize(0);
  HSideBand3DZY->GetYaxis()->SetTitle("#varphi");
  HSideBand3DZY->GetXaxis()->SetTitle("Cos#theta_{k}");
  HSideBand3DZY->Draw("LEGO2 0 fbbb");
//  cyz->cd(2);
//  HSBFuncZY->Scale(HSideBand3DZY->Integral()/HSBFuncZY->Integral()*NFact);
  HSBFuncZY->SetLineColor(kRed);
  HSBFuncZY->GetYaxis()->SetTitle("#varphi");
  HSBFuncZY->GetXaxis()->SetTitle("Cos#theta_{k}");
  HSBFuncZY->Draw("SURF A same fbbb");
//
  cxz->cd();
  gPad->SetTheta(40.);
  gPad->SetPhi(40.);
  //HSideBand3DZX->Rebin2D(2,2);
  HSideBand3DZX->Smooth(2);
  HSideBand3DZX->GetXaxis()->SetLabelSize(0);
  HSideBand3DZX->GetYaxis()->SetLabelSize(0);
  HSideBand3DZX->GetZaxis()->SetLabelSize(0);
  HSideBand3DZX->GetYaxis()->SetTitle("#varphi");
  HSideBand3DZX->GetXaxis()->SetTitle("Cos#theta_{l}");
  HSideBand3DZX->Draw("LEGO2 0 fbbb");
//  cxz->cd(2);
//  HSBFuncZX->Scale(HSideBand3DZX->Integral()/HSBFuncZX->Integral()*NFact);
  HSBFuncZX->SetLineColor(kRed);
  HSBFuncZX->GetYaxis()->SetTitle("#varphi");
  HSBFuncZX->GetXaxis()->SetTitle("Cos#theta_{l}");
  HSBFuncZX->Draw("SURF A same fbbb");

  
  
 
  
  
//
// Mass Spectrum
//
//   GooFit::Variable mean  ("mean"  ,5.2762,XStepMinuit, 5., 5.5);
//   GooFit::Variable sigma1("sigma1",0.0139,XStepMinuit, 0., 1.);
//   GooFit::Variable sigma2("sigma2",0.0228,XStepMinuit, 0., 1.);
//   GooFit::Variable sigma3("sigma3",0.0601,XStepMinuit, 0., 1.);
//////////////////////////////////
/// MASS ONLY                  ///
//////////////////////////////////
  cmass->cd();
  gStyle->SetTitleBorderSize(0);
  TH1D* HxMassQ2Clone  =(TH1D*)  HxMassQ2->Clone("");
  TH1D* HxMassQ2SBClone=(TH1D*)  HxMassQ2SB->Clone("");
  TH1D* pdfHxMassQ2Clone=(TH1D*) pdfHxMassQ2->Clone("");
  TH1D* sigHxMassQ2Clone=(TH1D*) sigHxMassQ2->Clone("");
  TH1D* bkgHxMassQ2Clone=(TH1D*) bkgHxMassQ2->Clone("");
  HxMassQ2Clone   ->GetXaxis()->SetRangeUser(HistMassL1,HistMassL2);
  HxMassQ2SBClone ->GetXaxis()->SetRangeUser(HistMassL1,HistMassL2);
  pdfHxMassQ2Clone->GetXaxis()->SetRangeUser(HistMassL1,HistMassL2);
  sigHxMassQ2Clone->GetXaxis()->SetRangeUser(HistMassL1,HistMassL2);
  bkgHxMassQ2Clone->GetXaxis()->SetRangeUser(HistMassL1,HistMassL2);
  HxMassQ2Clone   ->GetXaxis()->SetRangeUser(HistMassL1,HistMassL2);
  HxMassQ2SBClone ->GetXaxis()->SetRangeUser(XMinSBL,XMaxSBR);
  pdfHxMassQ2Clone->GetXaxis()->SetRangeUser(XMinSBL,XMaxSBR);
  sigHxMassQ2Clone->GetXaxis()->SetRangeUser(XMinSBL,XMaxSBR);
  bkgHxMassQ2Clone->GetXaxis()->SetRangeUser(XMinSBL,XMaxSBR);
//  pdfHxMassQ2->SetLineStyle(2);
//  sigHxMassQ2Clone->SetLineStyle(kDashed);
//  bkgHxMassQ2Clone->SetLineStyle(kDashed);
  TLegend* leg_signSBMass = new TLegend(0.53,0.65,0.65,0.88);
  leg_signSBMass->SetTextSize(0.022) ;
  leg_signSBMass->SetTextAlign(13);
  leg_signSBMass->SetBorderSize(0.);
  leg_signSBMass->SetFillStyle(0);
  leg_signSBMass->AddEntry(HxMassQ2Clone ,Form( "#color[2]{q^{2} bin  %d [%2.1f<q^{2}<%2.1f GeV^{2}/c^{4}]:}", Q2Bin, Q2Min,Q2Max),"");
  leg_signSBMass->AddEntry(HxMassQ2Clone ,Form( "Mass [data]"),"lep");
  leg_signSBMass->AddEntry(pdfHxMassQ2Clone ,Form( "fit model"),"l");
  leg_signSBMass->AddEntry(sigHxMassQ2Clone ,Form( "signal model"),"l");
  leg_signSBMass->AddEntry(bkgHxMassQ2Clone ,Form( "bckg model"),"l");
  if(SigmaProbSign==0){
   leg_signSBMass->AddEntry(HxMassQ2Clone ,Form( "sb events = %4.0f [-%s#sigma,-%1s#sigma]&[%s#sigma,%s#sigma]", \\
   SBEntries,NSigma1L,NSigma2L,NSigma1R,NSigma2R,FMTNSigma1L,FMTNSigma2L,FMTNSigma1R,FMTNSigma2R),"");
   leg_signSBMass->AddEntry(HxMassQ2Clone ,Form( "bckg events =%4.0f [-2#sigma,2#sigma]",NBckgInt2Sigma),"");
  }else{
   leg_signSBMass->AddEntry(HxMassQ2Clone ,Form( "sb entries=%4.0f",SBEntries),"");
   leg_signSBMass->AddEntry(HxMassQ2Clone ,Form( "bckg events =%4.0f [95.5%% of signal]",NBckgInt2Sigma),"");
  } 
//  leg_signSBMass->AddEntry(HxMassQ2Clone ,Form( "#chi^{2}_{/NDOF}=%5.2f NDOF=%d", AdaptChi2/NDegreeofFreedomAdapt,NDegreeofFreedomAdapt),"pel");
// //  leg_signSBMass->SetHeader(Form( "B_{0} #rightarrow K^{*0}#mu^{+}#mu^{-} Mass Spectrum for bin %d [%2.1f<q^{2}<%2.1f] ", Q2CloneBin, Q2CloneMin,Q2CloneMax));
//   leg_signSBMass->AddEntry(HxMassQ2Clone ,Form( "B_{0} #rightarrow K^{*0}#mu^{+}#mu^{-} Mass Spectrum for bin %d [%2.1f<q^{2}<%2.1f] ", Q2CloneBin, Q2CloneMin,Q2CloneMax),"pel");
//   leg_signSBMass->AddEntry(HxMassQ2CloneSB ,Form( "B_{0} #rightarrow K^{*0}#mu^{+}#mu^{-} Mass Spectrum for bin %d [%2.1f<q^{2}<%2.1f] ", Q2CloneBin, Q2CloneMin,Q2CloneMax),"pel");
//  gStyle->SetTitleFontSize(0.09) ;
  HxMassQ2Clone->SetTitle(Form("B_{0} #rightarrow K^{*0}#mu^{+}#mu^{-} Mass - q^{2} bin %d Run II %d [%2.1f<q^{2}<%2.1f Gev^{2}/c^{4}]",Q2Bin, RunEra, Q2Min,Q2Max));
//  HxMassQ2Clone->SetTitle(Form( "B_{0} #rightarrow K^{*0}#mu^{+}#mu^{-} Mass Spectrum and sideband for q^{2} bin %d [%2.1f<q^{2}<%2.1f] ", Q2CloneBin, Q2CloneMin,Q2CloneMax));
  HxMassQ2Clone->SetLineColor(kBlue);
  HxMassQ2Clone->GetXaxis()->SetTitle("Mass  (GeV/c^{2})");
  HxMassQ2Clone->GetYaxis()->SetTitleOffset(1.4);
  HxMassQ2Clone->GetYaxis()->SetTitle(Form("Events/(%4.4f GeV/c^{2})",(XMaxSign-XMinSign)/xMassHBin2));
  HxMassQ2Clone->DrawClone("E1");
//  HxMassQ2CloneSB->SetTitleSize(20);
  HxMassQ2SBClone->SetLineColor(kRed);
  HxMassQ2SBClone->SetTitle(Form( "B_{0} #rightarrow K^{*0}#mu^{+}#mu^{-} Mass Spectrum and sideband for q^{2} bin %d [%2.1f<q^{2}<%2.1f] ", Q2Bin, Q2Min,Q2Max));
  HxMassQ2SBClone->SetFillColor(kRed);
  HxMassQ2SBClone->DrawClone("SAME,B");
  pdfHxMassQ2Clone->DrawClone("same,HIST C");
//  SetHistLineStyle(2);
  sigHxMassQ2Clone->DrawClone("same,HIST C");
  bkgHxMassQ2Clone->DrawClone("same,HIST C");
  leg_signSBMass->Draw();
  cmass->Update();
  gStyle->SetHistLineStyle(1);
//////////////////////////////////
//////////////////////////////////
//////////////////////////////////

  OutFile->cd();
  covMatrix->Write();

  c1->cd();
//     TLegend* leg_sign = new TLegend(0.30,0.70,0.90,0.90);
//     leg_sign->SetTextSize(0.025) ;
//     leg_sign->SetTextAlign(31);
//     leg_sign->SetBorderSize(0.);
//     leg_sign->SetFillStyle(0);
//     leg_sign->SetHeader(Form( "B_{0} #rightarrow K^{*0}#mu^{+}#mu^{-} Mass Spectrum for bin %d [%2.1f<q^{2}<%2.1f] ", Q2Bin, Q2Min,Q2Max));
//     leg_sign->AddEntry(HxMass ,"","");
//     if(signalYield->getError()!=0){
//       leg_sign->AddEntry(&HxMass ,Form( "Yield_{Sign} =    %5.0f  #pm %5.0f",signalYield->getValue(),signalYield->getError()),"");
//     }else{
//       leg_sign->AddEntry(&HxMass ,Form( "Yield_{Sign} =    %5.0f Fixed",signalYield->getValue()),"");
//     }
//     if(bckgYield->getError()!=0){
//       leg_sign->AddEntry(&HxMass ,Form( "Yield_{Bckg} =    %5.0f  #pm  %5.0f",bckgYield->getValue(),bckgYield->getError()),"");
//     }else{
//       leg_sign->AddEntry(&HxMass ,Form( "Yield_{Bckg} =    %5.0f  Fixed",bckgYield->getValue()),"");
//     }
//     
//     if(mean.getError()!=0){
//      leg_sign->AddEntry(&HxMass ,Form( "M_{B^{0}} =   %5.5f  #pm %5.5f",mean.getValue(),mean.getError()),"");
//     }else{
//      leg_sign->AddEntry(&HxMass ,Form( "M_{B^{0}} =   %5.5f Fixed",mean.getValue()),"");
//      }
//     if(sigma1.getError()!=0){
//      leg_sign->AddEntry(&HxMass ,Form( "#sigma#scale[0.6]{1}_{B^{0}} =   %5.5f  #pm %5.5f",sigma1.getValue(),sigma1.getError()),"");
//     }else{
//      leg_sign->AddEntry(&HxMass ,Form( "#sigma#scale[0.6]{1}_{B^{0}} =   %5.5f Fixed",sigma1.getValue()),"");
//     }
//     if(sigma2.getError()!=0){
//      leg_sign->AddEntry(&HxMass ,Form( "#sigma#scale[0.6]{2}_{B^{0}} =   %5.5f  #pm %5.5f",sigma2.getValue(),sigma2.getError()),"");
//     }else{
//      leg_sign->AddEntry(&HxMass ,Form( "#sigma#scale[0.6]{2}_{B^{0}} =   %5.5f Fixed",sigma2.getValue()),"");
//    }
  gStyle->SetTitleBorderSize(0);
//  gStyle->SetTitleFontSize(0.08) ;
  HxMass->SetFillStyle(0);
  HxMass->SetTitle("B_{0} #rightarrow K^{*0}#mu^{+}#mu^{-} Mass Spectrum");
  HxMass->GetXaxis()->SetTitle("Mass (GeV/c^{2})");
  HxMass->SetMarkerStyle(8);
  HxMass->SetMarkerSize(MarkerSizeSet);
  HxMass->Draw("E1");
//   HxMassQ2->SetLineColor(kBlue);
//   HxMassQ2->Draw("same"); 

  HxMass->Write();
  HxMassQ2SB->Write();
  HxMassQ2->Write();
  HxMassVsCosL->Write();
  HxMassVsCosK->Write();
  HxMassVsPhi->Write();
  HxReco->Write();
  HSideBand3D->Write();
  HSBFunc->Write();
  HSBFuncX->Write();
  HSBFuncY->Write();
  HSBFuncZ->Write();
  HSBFuncXY->Write();
  HSBFuncZY->Write();
  HSBFuncZX->Write();
  HSideBand3DX->Write();
  HSideBand3DY->Write();
  HSideBand3DZ->Write();
//  HSideBand3DY_1->Write();
//  HSideBand3DZ_1->Write();
  HSideBand3DXY->Write();
  HSideBand3DZY->Write();
  HSideBand3DZX->Write();
  pdfHxMassQ2->Write();
  sigHxMassQ2->Write();
  bkgHxMassQ2->Write();
  
//   TH3D *H3Div = (TH3D *)HxReco->Clone(); 
//   H3Div->SetName("H3RecoDivGen");
//   H3Div->Divide(HxGene);
//
   cmass->Write();
   gSystem->Exec(Form("mv %s %s.tmp",PNGNameFitSB3DMass,PNGNameFitSB3DMass));
   cmass->Print(PNGNameFitSB3DMass);
   c1->Write();
   c6->Write();
   gSystem->Exec(Form("mv %s %s.tmp",PNGNameFitSB3D,PNGNameFitSB3D));
   c6->Print(PNGNameFitSB3D);
   gSystem->Exec(Form("mv %s %s.tmp",PDFNameFitSB3D,PDFNameFitSB3D));
   c6->Print(PDFNameFitSB3D);
   cxy->Write();
   gSystem->Exec(Form("mv %s %s.tmp",PNGNameProjXYHist,PNGNameProjXYHist));
   cxy->Print(PNGNameProjXYHist);
   cyz->Write();
   gSystem->Exec(Form("mv %s %s.tmp",PNGNameProjZYHist,PNGNameProjZYHist));
   cyz->Print(PNGNameProjZYHist);
   cxz->Write();
   gSystem->Exec(Form("mv %s %s.tmp",PNGNameProjZXHist,PNGNameProjZXHist));
   cxz->Print(PNGNameProjZXHist);
//   
   cprojX->Write();
   gSystem->Exec(Form("mv %s %s.tmp",PNGNameFitSB3DProjX,PNGNameFitSB3DProjX));
   cprojX->Print(PNGNameFitSB3DProjX);
//   
   cprojY->Write();
   gSystem->Exec(Form("mv %s %s.tmp",PNGNameFitSB3DProjY,PNGNameFitSB3DProjY));
   cprojY->Print(PNGNameFitSB3DProjY);
//
   cprojZ->Write();
   gSystem->Exec(Form("mv %s %s.tmp",PNGNameFitSB3DProjZ,PNGNameFitSB3DProjZ));
   cprojZ->Print(PNGNameFitSB3DProjZ);
//
   h2polxyContC->Write();
   h2polxzContC->Write();
   h2polxyDensC->Write();
   h2polxzDensC->Write();
   ca->Write();
   csignstudy->Write();
   
//   c7->Write();
//   c8->Write();
//   gSystem->Exec(Form("mv %s %s.tmp",PDFNameMass,PDFNameMass));
//   c1->Print(PDFNameMass);
//   sprintf(testo,"mv %s %s.tmp",PDFNameFitEffi3D,PDFNameFitEffi3D);
//   gSystem->Exec(Form("mv %s %s.tmp",PDFNameFitEffi3D,PDFNameFitEffi3D));
//   c6->Print(PDFNameFitEffi3D);
//   gSystem->Exec(Form("mv %s %s.tmp",PNGNameFitEffi3D,PNGNameFitEffi3D));
//   c6->Print(PNGNameFitEffi3D);
//   gSystem->Exec(Form("mv %s %s.tmp",PDFNameFitClosure,PDFNameFitClosure));
//   c8->Print(PDFNameFitClosure);
//   gSystem->Exec(Form("mv %s %s.tmp",PNGNameFitClosure,PNGNameFitClosure));
//   c8->Print(PNGNameFitClosure);
  OutFile->Close();
  
  std::cout<<"==========================================" <<std::endl;
  std::cout<<"==========================================" <<std::endl;
//=================================================================================  
//=================================================================================  
// START write par+norm txt files  
//=================================================================================  
//=================================================================================  
//   coeffy =0.0;
//   std::fstream *parListNormOutput =  new std::fstream(ListParNorm,ios::out);
//   std::fstream *parPlotNormOutput =  new std::fstream(ListPloNorm,ios::out);
//   if(parListNormOutput->is_open() && parPlotNormOutput->is_open() ){
//    std::cout<<"Open: "<<ListParNorm<<std::endl ;
//    std::cout<<"Open: "<<ListPloNorm<<std::endl ;
//    for(int i=0;i<numParameters;++i) {
//     if(fabs(coeffPoly[i].getValue())>fabs(coeffPoly[i].getError())){
//      coeffy=  coeffPoly[i].getValue();
//     }else{
//      coeffy=  0.0;
//     } 
//     *parListNormOutput <<std::scientific << std::setprecision(20)<< coeffy<<std::endl;
//     *parPlotNormOutput <<std::scientific << std::setprecision(20)<< coeffPoly[i].getValue()<<std::endl;
//    }
//
//    xCosL_x.setNumBins(SETNumBinsX);
//    xCosK_y.setNumBins(SETNumBinsY);
//    xPhiK_z.setNumBins(SETNumBinsZ);
//    cout<<"CosL integration NBins ="<<xCosL_x<<endl;
//    cout<<"CosK integration NBins ="<<xCosK_y<<endl;
//    cout<<"Phi  integration NBins ="<<xPhiK_z<<endl;
// //  
// // cout<<"cudaFreeHost(Norm)  = "<<(cudaFreeHost(model))<<endl;
//    cout<<"cudaFreeHost(modelPlot)  = "<<(cudaFreeHost(modelPlot))<<endl;
// //   cout<<"cudaFreeHost(model)      = "<<(cudaFreeHost(model))<<endl;
//    totalParams=0;
//    parListNormOutput->close();
//    parPlotNormOutput->close();
//    std::cout<<"Close: "<<ListParNorm<<std::endl ;
//    std::cout<<"Close: "<<ListPloNorm<<std::endl ;
//   }else{
//    if(!parListNormOutput->is_open()) std::cout<<"Error: can not open "<<ListParNorm<<std::endl ;
//    if(!parPlotNormOutput->is_open()) std::cout<<"Error: can not open "<<ListPloNorm<<std::endl ;
//    exit(1);
//   }
//=================================================================================  
//=================================================================================  
// END write par+norm txt files  
//=================================================================================  
//=================================================================================  
  std::cout<<"===================================================================="<<endl;
  std::cout<<"======== REMIND:  	NFact   ="<<NFact   <<"		=============="<<std::endl;
  std::cout<<"======== REMIND:  	NFactGen="<<NFactGen<<"		=============="<<std::endl;
  std::cout<<"====================================================================\n\n"<<endl;
  if(xCoeffIndex>0){
   std::cout<<"===================================================================="<<endl;
   std::cout<<Form("WARNING: Normalization could be fixed better if p(%d)=1",xCoeffIndex)<<std::endl ;
   std::cout<<"===================================================================="<<endl;
  }

  stopCPU = times(&stopProc);
  gettimeofday(&stopTime, NULL);
  // Print total minimization time
  double myCPU = stopCPU - startCPU;
  double totalCPU = myCPU; 

  timersub(&stopTime, &startTime, &totalTime);
  std::cout << "Wallclock time  : " << totalTime.tv_sec + totalTime.tv_usec/1000000.0 << " seconds." << std::endl;
  std::cout << "CPU time: " << (myCPU / CLOCKS_PER_SEC) << std::endl; 
  std::cout << "Total CPU time: " << (totalCPU / CLOCKS_PER_SEC) << std::endl; 
  myCPU = stopProc.tms_utime - startProc.tms_utime;
  std::cout << "Processor time: " << (myCPU / CLOCKS_PER_SEC) << std::endl;
  std::cout<<"==========================================" <<std::endl;
  std::cout<<"==========================================" <<std::endl;
}

//============================================================================================================================================================
//============================================================================================================================================================
//============================================================================================================================================================
//
//  CreateInputHistoFile();
//
//============================================================================================================================================================
//============================================================================================================================================================
//============================================================================================================================================================
void CreateInputHistoFile(){   

  if(Folded) printf("****************************** WARNING: Folded ******************************\n");
  
  TFile*OutFileNtupla = TFile::Open(OutFileNameInputHisto,"RECREATE");
  RecoB0TreeOut = new TTree(OutputRecoB0TreeName,OutputRecoB0TreeName) ;
  RecoB0TreeOut -> SetAutoSave(500000000);
  TCanvas* c2 = new TCanvas("c2","Fit Mass Spectrum",200,10,900,780);
  TCanvas* c3 = new TCanvas("c3","Reco Histograms",200,10,900,780);
  c3->Divide(2,2);  

  int nfile=0;
  TChain* RecoB0Tree = new TChain();  
  nfile = RecoB0Tree->Add(Form("%s/%s/%s",RecoDir,InputFileNameRecoB0,InputRecoB0TreeName));
  if( nfile==0 ||  !RecoB0Tree->GetFile() ){
    cout<<"Error:  no Reco files found!!!\n"<<endl;
    exit(1);
  }else{
    printf("Try to open %s/%s/%s\n",RecoDir,InputFileNameRecoB0,InputRecoB0TreeName);
    cout<<"Opening "<<nfile <<" Reco files found!!!\n"<<endl;
  }  
  if(!RecoB0Tree ){
    cout<<"TTree Reco Data: "<< InputRecoB0TreeName <<" not found!!!\n"<<endl;
    exit(1);
  }else{
    cout<<"TTree Reco Data: "<< InputRecoB0TreeName <<" OK FOUND!!!\n"<<endl;
  }  
  


  printf("(Mass Window     : xB0Mass>%8f && xB0Mass<%8f \n",XMinSign,XMaxSign);
//  printf("(SB Mass Windows : xB0Mass>%8f && xB0Mass<%8f || xB0Mass>%8f && xB0Mass<%8f)\n",XMinSBL,XMaxSBL,XMinSBR,XMaxSBR);




//   TH3D* HxReco = new   TH3D( "HxReco"    , "B^{0} Reco correct tagged",  xCosLHBin, XMinCosThetaL, XMaxCosThetaL,
//  									 xCosKHBin, XMinCosThetaK, XMaxCosThetaK,
// 									 xPhiHBin , XMinPhi, XMaxPhi );




	
						
//  GooFit::Observable* xcTau  = new GooFit::Observable("xcTau",XMin, XMax); 
  xMass.setNumBins( xMassHBin );      // set  step for integrals
  std::cout<<"xMass.getNumBins() = "<<xMass.getNumBins()<<std::endl;
  std::cout<<"Hist Mass Bin = "<<xMassHBin<<std::endl;
 

//======================================================================
//======================================================================
//======================================================================
//
//			      RECONSTRUCTED EVENTS
//
//======================================================================
//======================================================================
//======================================================================
  double  tagged_mass	 ;
  double  cos_theta_l	 ;
  double  cos_theta_k	 ;
  double  phi_kst_mumu   ;
  double  mumuMass	 ;
  double  mumuMassE	 ;
  double  recQ2 	 ;
  double  tagB0          ;
  double  mmk1		 ;
  double  mmk2		 ;
  double  bMass		 ;
  double  bBarMass	 ;
  double  dR_mum_trkm    ;
  double  dR_mup_trkp    ;
  bool    passB0Psi_lmnr ;
  bool    passB0Psi_jpsi ;
  bool    passB0Psi_psip ;
  bool    passB0Psi      ;
//  
  bool    XCut= false    ;
  double  kaonPt	=0;
  double  pionPt	=0;
  double  mmpiMass	=0;
  double  mmkMass	=0;
  double  wt_mass	=0;
  double  wt_kstarmass  =0;
//
  RecoB0Tree->SetBranchAddress("tagged_mass"   ,&tagged_mass);
  RecoB0Tree->SetBranchAddress("cos_theta_l"   ,&cos_theta_l);
  RecoB0Tree->SetBranchAddress("cos_theta_k"   ,&cos_theta_k);
  RecoB0Tree->SetBranchAddress("phi_kst_mumu"  ,&phi_kst_mumu);
  RecoB0Tree->SetBranchAddress("mumuMass"      ,&mumuMass);
  RecoB0Tree->SetBranchAddress("mumuMassE"     ,&mumuMassE);
  RecoB0Tree->SetBranchAddress("tagB0"         ,&tagB0);
  RecoB0Tree->SetBranchAddress("mmk1"          ,&mmk1);
  RecoB0Tree->SetBranchAddress("mmk2"          ,&mmk2);
  RecoB0Tree->SetBranchAddress("bMass"         ,&bMass);
  RecoB0Tree->SetBranchAddress("bBarMass"      ,&bBarMass);
  RecoB0Tree->SetBranchAddress("dR_mum_trkm",  &dR_mum_trkm);
  RecoB0Tree->SetBranchAddress("dR_mup_trkp",  &dR_mup_trkp);
  RecoB0Tree->SetBranchAddress("passB0Psi_lmnr",&passB0Psi_lmnr);
  RecoB0Tree->SetBranchAddress("passB0Psi_jpsi",&passB0Psi_jpsi);
  RecoB0Tree->SetBranchAddress("passB0Psi_psip",&passB0Psi_psip);
  RecoB0Tree->SetBranchAddress("kaonPt"		    ,&kaonPt	    );
  RecoB0Tree->SetBranchAddress("pionPt"		    ,&pionPt	    );
  RecoB0Tree->SetBranchAddress("mmpiMass" 	    ,&mmpiMass      );
  RecoB0Tree->SetBranchAddress("mmkMass"  	    ,&mmkMass	    );
  RecoB0Tree->SetBranchAddress("wt_mass"  	    ,&wt_mass	    );
  RecoB0Tree->SetBranchAddress("wt_kstarmass"	    ,&wt_kstarmass  );
  
  RecoB0TreeOut->Branch("cos_theta_l"   ,&cos_theta_l    ,   "cos_theta_l/D"   );
  RecoB0TreeOut->Branch("cos_theta_k"   ,&cos_theta_k    ,   "cos_theta_k/D"   );
  RecoB0TreeOut->Branch("phi_kst_mumu"  ,&phi_kst_mumu   ,   "phi_kst_mumu/D"  );
  RecoB0TreeOut->Branch("tagged_mass"   ,&tagged_mass    ,   "tagged_mass/D"   );
  RecoB0TreeOut->Branch("mumuMass"      ,&mumuMass       ,   "mumuMass/D"      );
  RecoB0TreeOut->Branch("mumuMassE"     ,&mumuMassE      ,   "mumuMassE/D"     );
  RecoB0TreeOut->Branch("mmk1"          ,&mmk1           ,   "mmk1/D"          );
  RecoB0TreeOut->Branch("mmk2"          ,&mmk2           ,   "mmk2/D"     );
  RecoB0TreeOut->Branch("kaonPt"      	,&kaonPt 	 ,   "kaonPt/D"        );
  RecoB0TreeOut->Branch("pionPt"      	,&pionPt 	 ,   "pionPt/D"        );
  RecoB0TreeOut->Branch("mmpiMass"    	,&mmpiMass       ,   "mmpiMass/D"      );
  RecoB0TreeOut->Branch("mmkMass"     	,&mmkMass        ,   "mmkMass/D"       );
  RecoB0TreeOut->Branch("wt_mass"     	,&wt_mass        ,   "wt_mass/D"       );
  RecoB0TreeOut->Branch("wt_kstarmass"	,&wt_kstarmass   ,   "wt_kstarmass/D"  );
//s
  double x0Cut=-0.4;
  double y0Cut= 0.3;
  double x1Cut= 0.6;
  double y1Cut=-0.1;
//  
  double x_0Cut=3;
  double y_0Cut=3.8;
  double x_1Cut=3.6;
  double y_1Cut=4.8;
  
  double CutX1=3.2;
  double CutX2=3.6;
  double CutY1=4.7;
  double CutY2=4.9;
//
//  double nSigma_psiRej =3.;
//
  std::vector<GooFit::Observable> dataMassVec;
  dataMassVec.push_back(xMass);
  UnbinnedDataSet* dataMass = new GooFit::UnbinnedDataSet(dataMassVec);
//  
//  
  int nentries = (int)RecoB0Tree->GetEntries();
  
   for (Int_t i=0;i<nentries;i++) { 
    RecoB0Tree->GetEntry(i);
    recQ2         =  mumuMass*mumuMass  ;
    if(Q2Bin==4){
    XCut=(((BpMass-wt_mass)-y0Cut)/(y1Cut-y0Cut))<(((wt_kstarmass-KstarMass)-x0Cut)/(x1Cut-x0Cut))&&kaonPt>pionPt&&(wt_kstarmass-KstarMass)>0
    	 &&(mmpiMass>CutX1&&mmpiMass<CutX2)&&(mmkMass>CutY1&&mmkMass<CutY2)&&((mmkMass-y_0Cut)/(y_1Cut-y_0Cut))>((mmpiMass-x_0Cut)/(x_1Cut-x_0Cut));
    passB0Psi =passB0Psi_jpsi&&!XCut;
    }else if(Q2Bin==6){  
     passB0Psi =passB0Psi_psip;
    }else{  
     passB0Psi =passB0Psi_lmnr;
    }  
//    double theBMass = tagged_mass;
//    double theBMass = tagB0*bMass+(1.-tagB0)*bBarMass;
//    double deltaB0M = theBMass-B0Mass;
//    double deltaJpsiM = mumuMass - JPsiMass;
//    double deltaPsiPM = mumuMass - PsiPMass;
//     if (!(recQ2> 8.68&&recQ2<10.09)&&
//         !(recQ2>12.86&&recQ2<14.18)){
	
//      if     ( fabs(mumuMass - JPsiMass) > nSigma_psiRej*mumuMassE && fabs(mumuMass - PsiPMass) > nSigma_psiRej*mumuMassE &&  \
//            (( mumuMass < JPsiMass && !( fabs(deltaB0M - deltaJpsiM) < 0.18 || fabs(deltaB0M - deltaPsiPM) < 0.0) ) || \
//             ( mumuMass > PsiPMass  && !( fabs(deltaB0M - deltaJpsiM) < 0.0  || fabs(deltaB0M - deltaPsiPM) < 0.09) ) || \
//             ( mumuMass > JPsiMass && mumuMass < PsiPMass && !( fabs(deltaB0M - deltaJpsiM) < 0.08 || fabs(deltaB0M - deltaPsiPM) < 0.08 )))){
//       if (! ( (mmk1 > 5.158 && mmk1 < 5.398) || (mmk2 > 5.158 && mmk2 < 5.398)) ){
//uffaaa

//        if ( fabs(mumuMass - JPsiMass) > nSigma_psiRej*mumuMassE && fabs(mumuMass - PsiPMass) > nSigma_psiRej*mumuMassE &&  \
//               (( mumuMass < JPsiMass && !( fabs(deltaB0M - deltaJpsiM) < 0.18 || fabs(deltaB0M - deltaPsiPM) < 0.0) ) || \
//                ( mumuMass > PsiPMass  && !( fabs(deltaB0M - deltaJpsiM) < 0.0  || fabs(deltaB0M - deltaPsiPM) < 0.08) ) || \
//                ( mumuMass > JPsiMass && mumuMass < PsiPMass && !( fabs(deltaB0M - deltaJpsiM) < 0.08 || fabs(deltaB0M - deltaPsiPM) < 0.09
//    	    )))){

//  if ( fabs(mumuMass - JPsiMass) > nSigma_psiRej*mumuMassE && fabs(mumuMass - PsiPMass) > nSigma_psiRej*mumuMassE &&  \
//         (( mumuMass < JPsiMass && !( fabs(deltaB0M - deltaJpsiM) < 0.16 || fabs(deltaB0M - deltaPsiPM) < 0.06) ) || \
//          ( mumuMass > PsiPMass && !( fabs(deltaB0M - deltaJpsiM) < 0.06 || fabs(deltaB0M - deltaPsiPM) < 0.03) ) || \
//          ( mumuMass > JPsiMass && mumuMass < PsiPMass && !( fabs(deltaB0M - deltaJpsiM) < 0.06 || fabs(deltaB0M - deltaPsiPM) < 0.06
//	    )))){
       if(passB0Psi){
        if(tagged_mass>=XMinSign&&tagged_mass<=XMaxSign){
//	if(dR_mum_trkm>0.0001&&
//	   dR_mup_trkp>0.0001
//	  ){
//
         xMass.setValue(tagged_mass);
         HxMass  ->Fill(tagged_mass);
         dataMass->addEvent();
         if(recQ2>Q2Min&&recQ2<Q2Max){
     	  if(cos_theta_l>=XMinCosThetaL&&cos_theta_l<=XMaxCosThetaL&&cos_theta_k>=XMinCosThetaK&&cos_theta_k<=XMaxCosThetaK){
     	       RecoB0TreeOut->Fill();
//
     	  }
     	 }
        }
       }
//      } 
//     }
//     }
    }
//   }
  
   
   
//  char TXT[200];
//  sprintf(TXT,"Mass Reco   Entries = %7f",HxMassQ2->GetEntries());
  cout<<"***********************************"<<endl;
  cout<<"***** RECONSTRUCTED EVENTS ********\n"<<endl;
  cout<<"***********************************\n"<<endl;
  cout<<"RecoB0Tree   Entries      = "<<nentries<<endl;
//  cout<<TXT<<endl;
  cout<<"\n***********************************"<<endl;
  cout<<"***********************************"<<endl;
//
  B0Sigma = FitMassSpectrum(dataMass, c2, HxMass,pdfHxMass,sigHxMass,bkgHxMass,5);
//  
  
  
 


  
//   TH1D* HxRecoX  = (TH1D*) HxReco->ProjectionX("HxRecoX",1,HxReco->GetNbinsY(),1,HxReco->GetNbinsZ());HxRecoX->SetTitle("HxReco Projection Cos#theta_{L}");
//   TH1D* HxRecoY  = (TH1D*) HxReco->ProjectionY("HxRecoY",1,HxReco->GetNbinsX(),1,HxReco->GetNbinsZ());HxRecoY->SetTitle("HxReco Projection Cos#theta_{K}");
//   TH1D* HxRecoZ  = (TH1D*) HxReco->ProjectionZ("HxRecoZ",1,HxReco->GetNbinsX(),1,HxReco->GetNbinsY());HxRecoZ->SetTitle("HxReco Projection #phi");
  
  OutFileNtupla->cd();
  
  c2->Write();
  gSystem->Exec(Form("mv %s %s.tmp",PNGNameMassHist,PNGNameMassHist));
//   HxReco->Write();
//   HxRecoX->Write();
//   HxRecoY->Write();
//   HxRecoZ->Write();
  HxMass->Write();
  pdfHxMass->Write();
  sigHxMass->Write();
  bkgHxMass->Write();
  HxMassQ2->Write();
  RecoB0TreeOut->Write();
  OutFileNtupla->Close();
  
//   sprintf(testo,"mv %s %s.tmp",PDFNameRecoHisto,PDFNameRecoHisto);
//   gSystem->Exec(testo);
//   c3->cd(1);HxRecoX->Draw();
//   c3->cd(2);HxRecoY->Draw();
//   c3->cd(3);HxRecoZ->Draw();
//   c3->Print(PDFNameRecoHisto);
  cout<<"**********************************************************************\n"<<endl;
  cout<<"save  HxReco  in "<<OutFileNameInputHisto<<"\n"<<endl;
  cout<<"**********************************************************************\n"<<endl;
}
//==========================================================================================
//
//       FitMassSpectrumRoofit
//
//==========================================================================================
double FitMassSpectrumRoofit(RooDataSet* data, TCanvas* c2, TH1D* masHist, TH1D* pdfHist, TH1D*sigHist, TH1D* bkgHist, int MaxDegreeBckg){
// //
//   if(MaxDegreeBckg<=0) {
//    cout<<"**********************************************************************\n"<<endl;
//    cout<<"Error!! MaxDegree <=0 in   FitMassSpectrumRoofit		       *\n"<<endl;
//    cout<<"**********************************************************************\n"<<endl;
//   }
// //
//     int Q2BinTMP=Q2Bin;
//      if(Q2Bin==6) {
//       Q2Bin=4;
//     }; 

    TFile* fitMassFile = new TFile( fitMassFileName, "READ" );
    if ( !fitMassFile || !fitMassFile ->IsOpen() ) {
      cout<<Form("File not found: %s\n",fitMassFileName)<<endl;
      exit(1);
    }
     RooWorkspace* w = (RooWorkspace*)fitMassFile->Get("w");
    if ( !w || w->IsZombie() ) {
     cout<<Form("Workspace not found in file:%s\n",fitMassFileName)<<endl;
     exit(1);
    } else {
     cout<<Form("Workspace Found!!! In file : %s\n",fitMassFileName)<<endl;
    }
//
    if(!(w->loadSnapshot(Form("reference_fit_RT_%d",Q2Bin)))){
      cout<<Form("Snapshot %s Workspace not found!!!\n",Form("reference_fit_RT_%d",Q2Bin))<<endl;
      exit(1);
    }else{
      w->loadSnapshot(Form("reference_fit_RT_%d",Q2Bin));
//      w->cd(Form("reference_fit_RT_%d",Q2Bin));
      cout<<Form("Snapshot %s Workspace found...\n",Form("reference_fit_RT_%d",Q2Bin))<<endl;
      cout<<"=========================================================================="<<endl;
      cout<<"=========================================================================="<<endl;
      cout<<Form(" DUMP WORKSPACE IN reference_fit_RT_%d",Q2Bin)<<endl;
      cout<<"=========================================================================="<<endl;
      cout<<Form("Snapshot %s Workspace found...\n",Form("reference_fit_RT_%d",Q2Bin))<<endl;
      w->Print("V");
      cout<<"=========================================================================="<<endl;
      cout<<"=========================================================================="<<endl;
    };

    w->ls();
    RooRealVar* tagged_mass= w->var("tagged_mass");
    double tagged_mass_rangeValMin=tagged_mass->getMin();
    double tagged_mass_rangeValMax=tagged_mass->getMax();
    std::cout<<Form("From workspace read tagged_mass in range [%f-%f]\n",tagged_mass_rangeValMin,tagged_mass_rangeValMax) <<std::endl;
    if( (tagged_mass_rangeValMin!=tagged_mass_rangeMin) || (tagged_mass_rangeValMax!=tagged_mass_rangeMax) ){
      std::cout<<Form("Warning! Force setting tagged_mass in range [%f-%f]\n",tagged_mass_rangeMin,tagged_mass_rangeMax) <<std::endl;
     tagged_mass->setRange(tagged_mass_rangeMin,tagged_mass_rangeMax);
    } 
//    tagged_mass->setRange(XMinSign,XMaxSign);
//    if(Q2Bin==0)  tagged_mass_rangeMin = 4.9;
    tagged_mass->setRange("full",tagged_mass_rangeMin,tagged_mass_rangeMax);
   
//    tagged_mass->setRange(XMinFull,XMaxFull);
    
//     RooRealVar *nsig_ref    = w->var("Yield");
//     RooRealVar *nbkg_ref    = w->var("nbkg");
// //
//     if(!nsig_ref){
//       cout<<"Yield from ref. fit  not found!!!\n"<<endl;
//       exit(1);
//     }else{
//       cout<<Form("Yield from ref. fit found = %f\n",nsig_ref->getVal())<<endl;
//     }
//     if(!nbkg_ref){
//       cout<<"BackYield form ref not found!!!\n"<<endl;
//       exit(1);
//     }else{
//       cout<<Form("BackYield  found = %f\n",nbkg_ref->getVal())<<endl;
//     }
   
     
    RooRealVar *yield_fromMC_RT    = w->var(Form("nRT_%d",Q2Bin));
    RooRealVar *yield_fromMC_WT    = w->var(Form("nWT_%d",Q2Bin));
//
    if(!yield_fromMC_RT){
      cout<<"yield_fromMC_RT  not found!!!\n"<<endl;
      exit(1);
    }else{
      cout<<Form("yield_fromMC_RT  found = %f\n",yield_fromMC_RT->getVal())<<endl;
    }
    if(!yield_fromMC_WT){
      cout<<"yield_fromMC_WT  not found!!!\n"<<endl;
      exit(1);
    }else{
      cout<<Form("yield_fromMC_WT  found = %f\n",yield_fromMC_WT->getVal())<<endl;
    }
    RooRealVar * fraction = new RooRealVar("fraction","fraction",0.,1.);
    fraction->setVal(yield_fromMC_RT->getVal()/(yield_fromMC_RT->getVal()+yield_fromMC_WT->getVal()));
    fraction->setError(sqrt(
    pow(yield_fromMC_RT->getError()*yield_fromMC_WT->getVal()/(yield_fromMC_RT->getVal()+yield_fromMC_WT->getVal())/(yield_fromMC_RT->getVal()+yield_fromMC_WT->getVal()),2)+
    pow(yield_fromMC_WT->getError()*yield_fromMC_RT->getVal()/(yield_fromMC_RT->getVal()+yield_fromMC_WT->getVal())/(yield_fromMC_RT->getVal()+yield_fromMC_WT->getVal()),2)));
    cout<<Form("fraction mistagged = %f +/- %f\n",fraction->getVal(),fraction->getError())<<endl;
//    
    RooRealVar    * mean_rt      = 0;
    RooAbsPdf     *theRTgauss    = 0;
    RooRealVar	  * sigma_rt1	 = 0;
    RooRealVar	  * sigma_rt2	 = 0;
    RooGaussian   * c_sigma_rt1  = 0;
    RooGaussian   * c_sigma_rt2  = 0;
    RooGaussian   * c_mean_rt	 = 0;
    RooGaussian   * c_f1rt	 = 0;
    RooProdPdf	  * c_RTgauss	 = 0;
    RooRealVar	  * alpha_rt1	 = 0;
    RooRealVar	  * alpha_rt2	 = 0;
    RooRealVar	  * n_rt1	 = 0;
    RooRealVar	  * n_rt2	 = 0;
    RooRealVar	  * f1rt	 = 0;
    RooArgSet	  * c_vars	 = 0;
    RooGaussian   * c_alpha_rt1  = 0;
    RooGaussian   * c_alpha_rt2  = 0;
    RooGaussian   * c_n_rt1	 = 0;
    RooGaussian   * c_n_rt2	 = 0;
//    RooFormulaVar * deltaPeaks   = 0;
//    RooGaussian   * c_deltaPeaks = 0;
    RooArgList    * c_pdfs       = 0;
    RooArgList    * c_pdfs_rt    = 0;
    RooArgList    * c_pdfs_wt    = 0;
    RooRealVar	  *mean_wt	 = 0;
    RooRealVar	  *sigma_wt	 = 0;
    RooRealVar	  *alpha_wt1	 = 0;
    RooRealVar	  *alpha_wt2	 = 0;
    RooRealVar	  *n_wt1	 = 0;
    RooRealVar	  *n_wt2	 = 0;




   if(w->pdf(Form("gauscb_RT_%d",Q2Bin))){ 
       theRTgauss   = w->pdf(Form("gauscb_RT_%d",Q2Bin));
       cout<<Form("FitMassSpectrumRoofit: gauscb_RT%d",Q2Bin)<<endl;
       mean_rt     = w->var(Form("mean_{RT}^{%d}",Q2Bin));
       sigma_rt1   = w->var(Form("#sigma_{RT1}^{%d}",Q2Bin));
       alpha_rt1   = w->var(Form("#alpha_{RT1}^{%d}",Q2Bin));
       n_rt1	   = w->var(Form("n_{RT1}^{%d}",Q2Bin));
       f1rt	   = w->var(Form("f^{RT%d}",Q2Bin));
       sigma_rt2   = w->var(Form("#sigma_{RT2}^{%d}",Q2Bin));
       c_sigma_rt1 = _constrainVar(sigma_rt1,w);
       c_alpha_rt1 = _constrainVar(alpha_rt1,w);
       c_sigma_rt2 = _constrainVar(sigma_rt2,w);
       c_f1rt	   = _constrainVar(f1rt,w);
       c_n_rt1     = _constrainVar(n_rt1,w);
       c_pdfs	 = new RooArgList(              *c_sigma_rt1, *c_sigma_rt2, *c_alpha_rt1, *c_n_rt1,*c_f1rt);
       c_pdfs_rt = new RooArgList(*theRTgauss,  *c_sigma_rt1, *c_sigma_rt2, *c_alpha_rt1, *c_n_rt1,*c_f1rt);
       c_vars	 = new RooArgSet(                 *sigma_rt1,   *sigma_rt2,   *alpha_rt1,   *n_rt1,*f1rt);
    } 
    
    
    
// RT Double Gaussian   
    if(w->pdf(Form("doublegaus_RT%d",Q2Bin))){
       cout<<Form("FitMassSpectrumRoofit: doublegaus_RT%d",Q2Bin)<<endl;
       theRTgauss   = w->pdf(Form("doublegaus_RT%d",Q2Bin));
       mean_rt     = w->var(Form("mean^{RT%d}",Q2Bin));
       sigma_rt1   = w->var(Form("#sigma_{1}^{RT%d}",Q2Bin));
       sigma_rt2   = w->var(Form("#sigma_{2}^{RT%d}",Q2Bin));
       f1rt	   = w->var(Form("f^{RT%d}",Q2Bin));
       c_sigma_rt1 = _constrainVar(sigma_rt1,w);
       c_sigma_rt2 = _constrainVar(sigma_rt2,w);
       c_mean_rt   = _constrainVar(mean_rt,w);
       c_f1rt	   = _constrainVar(f1rt,w);
 
       ////// creating constraints for the RT component
//       c_RTgauss = new RooProdPdf("c_RTgauss" , "c_RTgauss" , RooArgList(*theRTgauss,*c_sigma_rt1,*c_sigma_rt2,*c_mean_rt,*c_f1rt  ) );
//        c_pdfs    = new RooArgList(            *c_sigma_rt1,*c_sigma_rt2,*c_mean_rt,*c_f1rt);
//        c_pdfs_rt = new RooArgList(*theRTgauss,*c_sigma_rt1,*c_sigma_rt2,*c_mean_rt,*c_f1rt);
//        c_vars    = new RooArgSet(               *sigma_rt1,  *sigma_rt2,  *mean_rt ,*f1rt);
       c_pdfs    = new RooArgList(            *c_sigma_rt1,*c_sigma_rt2,*c_mean_rt,*c_f1rt);
       c_pdfs_rt = new RooArgList(*theRTgauss,*c_sigma_rt1,*c_sigma_rt2,*c_mean_rt,*c_f1rt);
       c_vars    = new RooArgSet(               *sigma_rt1,  *sigma_rt2,  *mean_rt,*f1rt);
    }
// RT Double CB   
    if( w->pdf(Form("doublecb_RT%d",Q2Bin))){ 
       cout<<Form("FitMassSpectrumRoofit: doublecb_RT%d",Q2Bin)<<endl;
       theRTgauss  = w->pdf(Form("doublecb_RT%d",Q2Bin));
       mean_rt     = w->var(Form("mean_{RT}^{%d}",Q2Bin));
       sigma_rt1   = w->var(Form("#sigma_{RT1}^{%d}",Q2Bin));
       alpha_rt1   = w->var(Form("#alpha_{RT1}^{%d}",Q2Bin));
       alpha_rt2   = w->var(Form("#alpha_{RT2}^{%d}",Q2Bin));
       n_rt1	   = w->var(Form("n_{RT1}^{%d}",Q2Bin));
       n_rt2	   = w->var(Form("n_{RT2}^{%d}",Q2Bin));
       f1rt	   = w->var(Form("f^{RT%d}",Q2Bin));
//      deltaPeaks  = RooFormulaVar("deltaPeaks%s"%ibin, "@0 - @1", RooArgList(mean_rt, mean_wt));  
       c_sigma_rt1 = _constrainVar(sigma_rt1,w);
       c_alpha_rt1 = _constrainVar(alpha_rt1,w);
       c_alpha_rt2 = _constrainVar(alpha_rt2,w);
       c_n_rt1     = _constrainVar(n_rt1,w);
       c_n_rt2     = _constrainVar(n_rt2,w);
       if (Q2Bin < 4){ 
           cout<<Form("FitMassSpectrumRoofit: fast doublecb_RT%d",Q2Bin)<<endl;
           c_pdfs    = new RooArgList(            *c_sigma_rt1, *c_alpha_rt1, *c_alpha_rt2, *c_n_rt1, *c_n_rt2);
           c_pdfs_rt = new RooArgList(*theRTgauss,*c_sigma_rt1, *c_alpha_rt1, *c_alpha_rt2, *c_n_rt1, *c_n_rt2);
           c_vars    = new RooArgSet(               *sigma_rt1,   *alpha_rt1,   *alpha_rt2,   *n_rt1,   *n_rt2);
       }else{
           cout<<Form("FitMassSpectrumRoofit: old doublecb_RT%d",Q2Bin)<<endl;
           sigma_rt2     = w->var(Form("#sigma_{RT2}^{%d}",Q2Bin));
           c_f1rt	 = _constrainVar(f1rt,w);
           c_sigma_rt2   = _constrainVar(sigma_rt2, w);
//            c_pdfs    = new RooArgList(              *c_sigma_rt1, *c_sigma_rt2, *c_alpha_rt1, *c_alpha_rt2, *c_n_rt1, *c_n_rt2, *c_f1rt);
//            c_pdfs_rt = new RooArgList(*theRTgauss , *c_sigma_rt1, *c_sigma_rt2, *c_alpha_rt1, *c_alpha_rt2, *c_n_rt1, *c_n_rt2, *c_f1rt);
//            c_vars    = new RooArgSet(                 *sigma_rt1,   *sigma_rt2,   *alpha_rt1,   *alpha_rt2,   *n_rt1,   *n_rt2, *c_f1rt);
           c_pdfs    = new RooArgList(              *c_sigma_rt1, *c_sigma_rt2, *c_alpha_rt1, *c_alpha_rt2, *c_n_rt1, *c_n_rt2);
           c_pdfs_rt = new RooArgList(*theRTgauss , *c_sigma_rt1, *c_sigma_rt2, *c_alpha_rt1, *c_alpha_rt2, *c_n_rt1, *c_n_rt2);
           c_vars    = new RooArgSet(                 *sigma_rt1,   *sigma_rt2,   *alpha_rt1,   *alpha_rt2,   *n_rt1,   *n_rt2);
       }
    } 

    if(!theRTgauss)  {
     cout<<"pdf theRTgauss not found!!!\n"<<endl;
     exit(1);
    } else{
     cout<<Form("pdf %s  found...\n",theRTgauss->GetName())<<endl;
    }
//
    c_RTgauss = new RooProdPdf("c_RTgauss" , "c_RTgauss" , *c_pdfs_rt );
//
    if(!(w->loadSnapshot(Form("reference_fit_WT_%d",Q2Bin)))){
      cout<<Form("Snapshot %s Workspace not found!!!\n",Form("reference_fit_WT_%d",Q2Bin))<<endl;
      exit(1);
    }else{
      w->loadSnapshot(Form("reference_fit_WT_%d",Q2Bin));
//      w->cd(Form("reference_fit_WT_%d",Q2Bin));
      
      cout<<"=========================================================================="<<endl;
      cout<<"=========================================================================="<<endl;
      cout<<Form(" DUMP WORKSPACE IN reference_fit_WT_%d",Q2Bin)<<endl;
      cout<<"=========================================================================="<<endl;
      cout<<Form("Snapshot %s Workspace found...\n",Form("reference_fit_WT_%d",Q2Bin))<<endl;
      w->Print("V");
      cout<<"=========================================================================="<<endl;
      cout<<"=========================================================================="<<endl;
    };
    
//    
    mean_wt	= !(w->var(Form("mean^{WT%d}",Q2Bin)))?(w->var(Form("mean_{WT}^{%d}",Q2Bin))):(w->var(Form("mean^{WT%d}",Q2Bin)));
    sigma_wt	= !(w->var(Form("#sigma_{CB}^{WT%d}",Q2Bin)))?(w->var(Form("#sigma_{WT1}^{%d}",Q2Bin))):(w->var(Form("#sigma_{CB}^{WT%d}",Q2Bin)));
    alpha_wt1	= !(w->var(Form("#alpha_{1}^{WT%d}",Q2Bin)))?(w->var(Form("#alpha_{WT1}^{%d}",Q2Bin))):(w->var(Form("#alpha_{1}^{WT%d}",Q2Bin)));
    alpha_wt2	= !(w->var(Form("#alpha_{2}^{WT%d}",Q2Bin)))?(w->var(Form("#alpha_{WT2}^{%d}",Q2Bin))):(w->var(Form("#alpha_{2}^{WT%d}",Q2Bin)));
    n_wt1	= !(w->var(Form("n_{1}^{WT%d}",Q2Bin)))?(w->var(Form("n_{WT1}^{%d}",Q2Bin))):(w->var(Form("n_{1}^{WT%d}",Q2Bin)));
    n_wt2	= !(w->var(Form("n_{2}^{WT%d}",Q2Bin)))?(w->var(Form("n_{WT2}^{%d}",Q2Bin))):(w->var(Form("n_{2}^{WT%d}",Q2Bin)));
//    
//  workaround
//    
//     if(Q2Bin==7){
//       n_wt2->setMax(150);
//     }  
      
//    
    RooAbsPdf *theWTgauss = w->pdf(Form("doublecb_%d",Q2Bin));
    if(!theWTgauss)  {
     cout<<"pdf theWTgauss not found!!!\n"<<endl;
     exit(1);
    } else{
     cout<<Form("pdf %s  found...\n",theWTgauss->GetName())<<endl;
    }
    RooGaussian* c_mean_wt     = _constrainVar(mean_wt, w);
    RooGaussian* c_sigma_wt    = _constrainVar(sigma_wt, w);
    RooGaussian* c_alpha_wt1   = _constrainVar(alpha_wt1, w);
    RooGaussian* c_alpha_wt2   = _constrainVar(alpha_wt2, w);
    RooGaussian* c_n_wt1       = _constrainVar(n_wt1, w);
    RooGaussian* c_n_wt2       = _constrainVar(n_wt2, w);
//   if(w->obj( "deltaPeaks")) {
//    if(w->obj( Form("deltaPeaks%d",Q2Bin))) {
//     deltaPeaks  = (RooFormulaVar *)w->obj(Form("deltaPeaks%d",Q2Bin));
//     c_deltaPeaks = new RooGaussian(Form("deltaPeaks%d",Q2Bin) , "c_deltaPeaks", *deltaPeaks, RooConst( deltaPeaks->getVal() ), RooConst( 0.0005 )); // value to be checked
// //    c_vars->add(*deltaPeaks);
// //    c_pdfs ->add(*c_deltaPeaks);
//    }else if(w->pdf(Form("doublecb_RT%d",Q2Bin))){
//     cout<<Form("FitMassSpectrumRoofit: deltaPeaks NOT FOUND!!!")<<endl;
// //    exit(0);
//    }	    

 //     RooGaussian* c_sigma_wt2   = _constrainVar(sigma_wt2, w);
//     RooGaussian* c_f3          = _constrainVar(f3wt, w);
    

    ////// creating constraints for the WT component
//    RooProdPdf* c_WTgauss  = new RooProdPdf("c_WTgauss" , "c_WTgauss"\
//    ,RooArgList(*theWTgauss,*c_alpha_wt1,*c_n_wt1,*c_sigma_wt,*c_mean_wt,*c_alpha_wt2,*c_n_wt2  ) );     
    RooRealVar  frt("F_{RT}"			  , "frt"   , fraction->getVal() , 0, 1);
    RooGaussian c_frt("c_frt"           	   , "c_frt" , frt,  RooFit::RooConst(fraction->getVal()) , RooFit::RooConst(fraction->getError()) );
//    RooAddPdf	signalFunction("sumgaus"	  , "rt+wt" , RooArgList(*theRTgauss,*theWTgauss), RooArgList(frt));
//     RooGaussian c_frt("c_frt"           	   , "c_frt" , frt,  (fraction) , (*fraction_s) );
//    RooGaussian c_frt("c_frt"           	   , "c_frt" , frt,  RooFit::RooConst(0.87628877977) , RooFit::RooConst(0.000523435458235) );
 
    c_pdfs_wt = new RooArgList(*theWTgauss);
    c_pdfs_wt->add(*c_sigma_wt);
    c_pdfs_wt->add(*c_mean_wt);
    c_pdfs_wt->add(*c_alpha_wt1);
    c_pdfs_wt->add(*c_alpha_wt2);
    c_pdfs_wt->add(*c_n_wt1);
    c_pdfs_wt->add(*c_n_wt2);


    RooProdPdf* c_WTgauss  = new RooProdPdf("c_WTgauss" , "c_WTgauss",*c_pdfs_wt);
    RooAddPdf	signalFunction("sumgaus"	  , "rt+wt" , RooArgList(*c_RTgauss,*c_WTgauss), RooArgList(frt));
    RooProdPdf  c_signalFunction("c_signalFunction", "c_signalFunction", RooArgList(signalFunction, c_frt))   ;  
    c_pdfs->add(*c_sigma_wt);
    c_pdfs->add(*c_mean_wt);
    c_pdfs->add(*c_alpha_wt1);
    c_pdfs->add(*c_alpha_wt2);
    c_pdfs->add(*c_n_wt1);
    c_pdfs->add(*c_n_wt2);
    c_pdfs->add(c_frt);
//    c_pdfs->add(signalFunction);

    c_vars->add(*sigma_wt);
    c_vars->add(*mean_wt);
    c_vars->add(*alpha_wt1);
    c_vars->add(*alpha_wt2);
    c_vars->add(*n_wt1);
    c_vars->add(*n_wt2);
    c_vars->add(frt);

////// now create background parametrization
    RooRealVar*  slope= new RooRealVar("slope"      , "slope"           ,    0.5,   -10, 10);
//    RooExponential bkg_exp("bkg_exp"    , "exponential"     ,  *slope,   *tagged_mass  );
//     RooRealVar     pol_c1("p1"          , "coeff x^0 term"  ,    0.5,   -10, 10);
//     RooRealVar     pol_c2("p2"          , "coeff x^1 term"  ,    0.5,   -10, 10);
//     RooRealVar     pol_c3("p3"          , "coeff x^2 term"  ,    0.5,   -10, 10);
//     RooRealVar     pol_c4("p4"          , "coeff x^3 term"  ,    0.5,   -10, 10);
// 
//     RooChebychev   bkg_exp("bkg_exp"    , "2nd order pol"   ,  *tagged_mass, RooArgList(pol_c1,pol_c2,pol_c3,pol_c4));

    double pol_bmax =1.;
    if(Q2Bin==6) pol_bmax =1.;
    RooRealVar*     pol_b0= new RooRealVar("pol_b0"          , "b0"  ,    pol_bmax  );
    RooRealVar*     pol_b1= new RooRealVar("pol_b1"          , "b1"  ,    0.1,  0., pol_bmax);
    RooRealVar*     pol_b2= new RooRealVar("pol_b2"          , "b2"  ,    0.1,  0., pol_bmax);
    RooRealVar*     pol_b3= new RooRealVar("pol_b3"          , "b3"  ,    0.0 );
    RooRealVar*     pol_b4= new RooRealVar("pol_b4"          , "b4"  ,    0.1 , 0., pol_bmax);
    if(Q2Bin!=4){
//   if(Q2Bin!=4&&Q2Bin!=6){
     bkg_exp = new RooExponential("bkg_exp"    , "exponential"     ,  *slope,   *tagged_mass  );
    }else{
//     pol_b0->setConstant(kTRUE);
//     pol_b4->setConstant(kTRUE);
//     pol_b3->setConstant(kTRUE);
     bkg_exp = new RooBernstein("bkg_exp"    , "bernstein pol"  ,  *tagged_mass, RooArgList(*pol_b0,*pol_b1,*pol_b2,*pol_b3,*pol_b4));
    }
    
    int NCPU=1;
    if(NFactGen>1) NCPU=10;
    double yieldIni = NFactGen*1000;
    double backgIni = NFactGen*1000;
    double yieldMin = 0.;
    double backgMin = 0.;
    double yieldMax = NFactGen*1000000.;
    double backgMax = NFactGen*1000000.;
   if(Q2Bin==4) {
       yieldIni = yieldSignal;
       backgIni = yieldBckg;
       yieldMin = 100000.;
       backgMin = 100000.;
       yieldMax = 2000000.;
       backgMax = 1000000.;
       NCPU=60;
    }   
   if(Q2Bin==6) {
       yieldIni = 90000;
       backgIni = 30000;
       yieldMin = 10000.;
       backgMin = 10000.;
       yieldMax = 200000.;
       backgMax = 100000.;
    }   
    RooRealVar     nsig("Yield"         , "signal frac"    ,   yieldIni,     yieldMin, yieldMax  );
    RooRealVar     nbkg("nbkg"          , "bkg fraction"   ,   backgIni,     backgMin, backgMax  );

//    RooRealVar     nsig("Yield"         , "signal frac"    ,    4000,     0,   1000000);
//    RooRealVar     nbkg("nbkg"          , "bkg fraction"   ,    1000,     0,   550000);
// 
//    RooProdPdf  c_signalFunction("c_signalFunction", "c_signalFunction", RooArgList(signalFunction, c_frt))   ;  
//    RooProdPdf c_signalFunction("c_signalFunction", "c_signalFunction", *c_pdfs);
    RooAddPdf fitFunction("fitfunction" , "fit function"  ,  RooArgList(c_signalFunction, *bkg_exp), RooArgList(nsig, nbkg));
//
//    RooAddPdf fitFunction("fitfunction" , "fit function"  ,  RooArgList(signalFunction, bkg_pol), RooArgList(nsig, nbkg));
//     tagged_mass->setRange("fullRedefined",XMinSBL,XMaxSBR);
    RooFitResult* r = fitFunction.fitTo(*data, 
    		       RooFit::Extended(kTRUE), 
    		       RooFit::NumCPU(NCPU),
    		       RooFit::Save(), 
    		       RooFit::Range("full"), 
    		       RooFit::Verbose(kFALSE),
    		       RooFit::Constrain(*c_vars)
    		      );
     		      
   r->Print();	
//   Q2Bin=Q2BinTMP;	      
//
//
// save a clone of bkg_exp 
//

    if(Q2Bin!=4){
//    if(Q2Bin!=4&&Q2Bin!=6){
     bkg_mass_sb = (RooExponential*)bkg_exp->clone(Form("bkg_mass_sb_bin%d_%d",Q2Bin,RunEra) );
     bkg_mass_sb->setNormRange("full");
    }else{ 
     bkg_mass_sb = (RooBernstein*)  bkg_exp->clone(Form("bkg_mass_sb_bin%d_%d",Q2Bin,RunEra) );
     bkg_mass_sb->setNormRange("full");
    } 
//   RooAbsBinning binning = (tagged_mass->getBinning("full")) ;
//     data->getRange(*tagged_mass,tagged_mass_rangeMin,tagged_mass_rangeMax);
//    std::cout<<Form("%f<[fit mass range]<%f",tagged_mass_rangeMin,tagged_mass_rangeMax)<<std::endl; 
//    tagged_mass_rangeMin=tagged_mass->getMin("full");
//    tagged_mass_rangeMax=tagged_mass->getMax("full");
//    std::cout<<Form("%f<[fit mass range]<%f",tagged_mass_rangeMin,tagged_mass_rangeMax)<<std::endl; 

//      cout<<"----"<<endl;
//       exit(1);
//      data->Print("V");
//      c2->cd();
//      RooPlot* frame = tagged_mass->frame( );
//      data->plotOn(frame, Binning(35), MarkerSize(.7));
//     fitFunction.plotOn(frame);
//     drawPdfComponents(fitFunction, frame, ROOT.kAzure, RooFit.NormRange("full"), RooFit.Range("full"), isData = True);
// 
//     fitFunction.paramOn(frame,  RooFit.Layout(0.62,0.86,0.88));
//     frame.Draw();
//     niceFrame(frame, '')
//     frame. addObject(_writeFitStatus(r))
// 
//     if not args.year=='test':  writeCMS(frame, args.year, [ q2binning[ibin], q2binning[ibin+1] ])
//     frame.Draw()
 //    c2->Print("test.pdf");
 
     double B0SigmaTemp=0.;
     c2->cd();
     TLegend* leg_sign = new TLegend(0.30,0.48,0.90,0.90);
     leg_sign->SetTextSize(0.025) ;
     leg_sign->SetTextAlign(31);
     leg_sign->SetBorderSize(0.);
     leg_sign->SetFillStyle(0);
     leg_sign->SetHeader("B^{0} mass spectrum  Fit Projection");
     if(nsig.getError()!=0){
       leg_sign->AddEntry(masHist ,Form( "Yield_{Sign} =     %5.0f  #pm %5.0f",nsig.getVal(),nsig.getError()),"");
     }else{
       leg_sign->AddEntry(masHist ,Form( "Yield_{Sign} =     %5.0f Fixed",nsig.getVal()),"");
     }
     if(nbkg.getError()!=0){
       leg_sign->AddEntry(masHist ,Form( "Yield_{Bckg} =     %5.0f  #pm  %5.0f",nbkg.getVal(),nbkg.getError()),"");
     }else{
       leg_sign->AddEntry(masHist ,Form( "Yield_{Bckg} =     %5.0f  Fixed",nbkg.getVal()),"");
     }
     if(mean_rt->getError()!=0){
      leg_sign->AddEntry(masHist ,Form( "M_{B^{0}}[RT] =   %5.5f  #pm %5.5f",mean_rt->getVal(),mean_rt->getError()),"");
     }else{
      leg_sign->AddEntry(masHist ,Form( "M_{B^{0}}[RT] =   %5.5f Fixed",mean_rt->getVal()),"");
      }
//     if(mean_wt==0){exit(0);};
     if(mean_wt->getError()!=0){
      leg_sign->AddEntry(masHist ,Form( "M_{B^{0}}[WT] =   %5.5f  #pm %5.5f",mean_wt->getVal(),mean_wt->getError()),"");
     }else{
      leg_sign->AddEntry(masHist ,Form( "M_{B^{0}}[WT] =   %5.5f Fixed",mean_wt->getVal()),"");
      }
     if(sigma_rt1->getError()!=XStepMinuit){
      leg_sign->AddEntry(masHist ,Form( "#sigma_{1}^{RT%d}#scale[0.6]{1}_{B^{0}} =   %5.5f  #pm %5.5f",Q2Bin,sigma_rt1->getVal(),sigma_rt1->getError()),"");
     }else{
      leg_sign->AddEntry(masHist ,Form( "#sigma_{1}^{RT%d}#scale[0.6]{1}_{B^{0}} =   %5.5f Fixed",Q2Bin,sigma_rt1->getVal()),"");
     }
     if(sigma_rt2!=0){
      if(sigma_rt2->getError()!=XStepMinuit){
      leg_sign->AddEntry(masHist ,Form( "#sigma_{2}^{RT%d}#scale[0.6]{2}_{B^{0}} =   %5.5f  #pm %5.5f",Q2Bin,sigma_rt2->getVal(),sigma_rt2->getError()),"");
//   	}else{
//    	 leg_sign->AddEntry(masHist ,Form( "#sigma#scale[0.6]{2}_{B^{0}} =   %5.5f Fixed",sigma2.getVal()),"");
      }
     } 
     if(sigma_wt->getError()!=XStepMinuit){
      leg_sign->AddEntry(masHist ,Form( "#sigma^{WT%d}#scale[0.6]{1}_{B^{0}} =   %5.5f  #pm %5.5f",Q2Bin,sigma_wt->getVal(),sigma_wt->getError()),"");
     }else{
      leg_sign->AddEntry(masHist ,Form( "#sigma^{WT%d}#scale[0.6]{1}_{B^{0}} =   %5.5f Fixed",Q2Bin,sigma_wt->getVal()),"");
     }
//      if(sigma_wt2){
//       if(sigma_wt2->getError()!=XStepMinuit){
//       leg_sign->AddEntry(masHist ,Form( "#sigma_{2}^{WT%d}#scale[0.6]{2}_{B^{0}} =   %5.5f  #pm %5.5f",Q2Bin,sigma_wt2->getVal(),sigma_wt2->getError()),"");
// //   	}else{
// //    	 leg_sign->AddEntry(masHist ,Form( "#sigma#scale[0.6]{2}_{B^{0}} =   %5.5f Fixed",sigma2.getVal()),"");
//       }
//     } 
     double min_CBGaus_rt=0;
     double max_CBGaus_rt=0;
//      float x1zoom=5.1;
//      float x2zoom=5.4;
     float x1zoom=XMinSign;
     float x2zoom=XMaxSign;
//      RooPlot *rframe = tagged_mass->frame(Title("Signal models RT"));
//      RooPlot *wframe = tagged_mass->frame(Title("Signal models WT"));
     RooAbsPdf *gaussRT_study = 0;
     RooGaussian *gaussRT_study1 = 0;
     RooGaussian *gaussRT_study2 = 0;
     TF1 * Func_theRTgauss    = 0;
     TF1 * Func_gaussRT_study = 0;
     TF1 * Clone_gaussRT_study = 0;
//     RooRealVar  f3("f3","f3",0.);
     if(w->pdf(Form("doublecb_RT%d",Q2Bin))){
      if(Q2Bin<4){
       min_CBGaus_rt =  mean_rt->getVal()-alpha_rt1->getVal()*sigma_rt1->getVal();
       max_CBGaus_rt =  mean_rt->getVal()+alpha_rt2->getVal()*sigma_rt1->getVal();
       gaussRT_study = new RooGaussian("gaussRT_study","gauss RT study"    ,*tagged_mass,*mean_rt,*sigma_rt1);
      }else{
//       sigma_rt2_pos = new RooRealVar("sigma_rt2_pos","|(sigma_rt2)|",fabs(sigma_rt2->getVal()));
       min_CBGaus_rt =  mean_rt->getVal()-fabs(alpha_rt1->getVal())*sigma_rt1->getVal();
       max_CBGaus_rt =  mean_rt->getVal()+fabs(alpha_rt2->getVal())*sigma_rt2->getVal();
       gaussRT_study1 = new RooGaussian("gaussRT_study1","gauss RT study1"    ,*tagged_mass,*mean_rt,*sigma_rt1);
       gaussRT_study2 = new RooGaussian("gaussRT_study2","gauss RT study2"    ,*tagged_mass,*mean_rt,*sigma_rt2);
       gaussRT_study = new RooAddPdf("gaussRT_study","gauss RT study"    ,RooArgList(*gaussRT_study1,*gaussRT_study2),RooArgList(*f1rt));
      } 
      std::cout<<Form("RT Double CB Gaus = %f<mass<%f",min_CBGaus_rt,max_CBGaus_rt)<<std::endl;
      csignstudy->cd(1);
      gPad->SetLeftMargin(0.15);
      Func_theRTgauss	 = theRTgauss	->asTF( RooArgList(*tagged_mass) );
      Func_gaussRT_study = gaussRT_study->asTF( RooArgList(*tagged_mass) );
      Func_theRTgauss->SetTitle("RT Model");
//      rframe->GetYaxis()->SetTitleOffset(1.4);
//      theRTgauss->plotOn(rframe,LineColor(kRed));
//      gaussRT_study->plotOn(rframe,LineColor(kBlue));
//      theRTgauss->plotOn(rframe, Range(min_CBGaus_rt,max_CBGaus_rt,kFALSE),FillColor(kRed),DrawOption("F"),FillStyle(3013),VLines());
      TLegend* leg_rt = new TLegend(0.70,0.70,0.90,0.90);
      leg_rt->SetTextSize(0.025) ;
      leg_rt->SetTextAlign(31);
      leg_rt->SetBorderSize(0.);
      leg_rt->SetFillStyle(0);
      leg_rt->AddEntry(Func_theRTgauss ,Form("#color[2]{Double CB Model}"),"");
      leg_rt->AddEntry(Func_gaussRT_study,Form("#color[4]{(2)Gaussian  Model}"),"");
//      rframe->addObject(leg_rt);
//      rframe->Draw();
      Func_theRTgauss->SetLineColor(kRed);
      Func_gaussRT_study->SetLineColor(kBlue);
      Func_theRTgauss->SetLineWidth(1.);
      Func_gaussRT_study->SetLineWidth(1.);
//     Func_gaussRT_study->SetLineStyle(kDashed);
      Func_theRTgauss->SetRange(x1zoom,x2zoom);
      Func_gaussRT_study->SetRange(x1zoom,x2zoom);
      Func_theRTgauss->Draw();
      Func_gaussRT_study->Draw("SAME");
      Clone_gaussRT_study =  (TF1*)Func_gaussRT_study->Clone();
      Clone_gaussRT_study->SetRange(min_CBGaus_rt,max_CBGaus_rt);
      Clone_gaussRT_study->SetFillColor(kBlue);
      Clone_gaussRT_study->SetFillStyle(3013);
      Clone_gaussRT_study->SetLineWidth(1.);
      Clone_gaussRT_study->Draw("SAME FC");
      leg_rt->Draw("SAME");
     } 
 //
     double min_CBGaus_wt =  mean_wt->getVal()-alpha_wt1->getVal()*sigma_wt->getVal();
     double max_CBGaus_wt =  mean_wt->getVal()+alpha_wt2->getVal()*sigma_wt->getVal();
     RooGaussian *gaussWT_study = new RooGaussian("gaussWT_study","gauss WT study"    ,*tagged_mass,*mean_wt,*sigma_wt);
     std::cout<<Form("WT Double CB Gaus = %f<mass<%f",min_CBGaus_wt,max_CBGaus_wt)<<std::endl;
     csignstudy->cd(2);
     gPad->SetLeftMargin(0.15);
//     wframe->GetYaxis()->SetTitleOffset(1.4);
//     theWTgauss->plotOn(wframe,LineColor(kRed));
     TF1 * Func_theWTgauss    = theWTgauss   ->asTF( RooArgList(*tagged_mass) );
     TF1 * Func_gaussWT_study = gaussWT_study->asTF( RooArgList(*tagged_mass) );
      Func_theWTgauss->SetTitle("WT Model");
//     RooAbsReal* IntegtheWTgauss    = theWTgauss->createIntegral(*tagged_mass,*tagged_mass,"full");
//     double scal = sigma_wt->getVal()*sqrt(2*TMath::Pi())*theWTgauss->getVal(RooArgList(*mean_wt));
//     cout<<IntegtheWTgauss->getVal()<<" "<<f->GetMaximum()<<endl;exit(0);
//     double scal = sigma_wt->getVal()*sqrt(2*TMath::Pi())*theWTgauss->getVal(RooArgList(*mean_wt));
//     double scal = IntegtheWTgauss->getVal()/gaussWT_study->getVal(RooArgList(*mean_wt));
//     gaussWT_study->plotOn(wframe,LineColor(kBlue),Normalization(1/IntegtheWTgauss->getVal()));
//     theWTgauss->plotOn(wframe, Range(min_CBGaus_wt,max_CBGaus_wt,kFALSE),FillColor(kRed),DrawOption("F"),FillStyle(3013),VLines());
     TLegend* leg_wt = new TLegend(0.70,0.70,0.90,0.90);
     leg_wt->SetTextSize(0.025) ;
     leg_wt->SetTextAlign(31);
     leg_wt->SetBorderSize(0.);
     leg_wt->SetFillStyle(0);
     leg_wt->AddEntry(Func_theWTgauss ,Form("#color[2]{Double CB Model}"),"");
     leg_wt->AddEntry(Func_gaussWT_study ,Form("#color[4]{Gaussian  Model}"),"");
//     leg_wt->AddEntry(theWTgauss ,Form("#color[2]{Double CB Model}"),"");
//     leg_wt->AddEntry(gaussWT_study ,Form("#color[4]{Gaussian  Model}"),"");
//     wframe->addObject(leg_wt);
//     wframe->Draw();
     Func_theWTgauss->SetLineColor(kRed);
     Func_gaussWT_study->SetLineColor(kBlue);
     Func_theWTgauss->SetLineWidth(1.0);
     Func_gaussWT_study->SetLineWidth(1.0);
//     Func_gaussWT_study->SetLineStyle(kDashed);
     Func_theWTgauss->SetRange(x1zoom,x2zoom);
     Func_gaussWT_study->SetRange(x1zoom,x2zoom);
     Func_theWTgauss->Draw();
     Func_gaussWT_study->Draw("SAME");
     TF1* Clone_gaussWT_study =  (TF1*)Func_gaussWT_study->Clone();
     Clone_gaussWT_study->SetRange(min_CBGaus_wt,max_CBGaus_wt);
     Clone_gaussWT_study->SetFillColor(kBlue);
     Clone_gaussWT_study->SetFillStyle(3013);
     Clone_gaussWT_study->SetLineWidth(1.0);
//     Clone_gaussWT_study->GetXaxis()->SetRangeUser(5.1,5.4);
     Clone_gaussWT_study->Draw("SAME FC");
     leg_wt->Draw("SAME");
     
     char PNGSignStudy[300]="";sprintf(PNGSignStudy,Form("signal-mass-study-Q2Bin-%d.png",Q2Bin));
     gSystem->Exec(Form("mv %s %s.tmp",PNGSignStudy,PNGSignStudy));
     csignstudy->Print(PNGSignStudy);
     
     double B0SigmaRT=0;
     double Sigma1RT =sigma_rt1->getVal();
     if(sigma_rt2!=0){
      double Sigma2RT =sigma_rt2->getVal();
      double WG1=f1rt->getVal();
      leg_sign->AddEntry(masHist ,Form( "f^{RT%d} =   %5.5f  #pm %5.5f",Q2Bin,f1rt->getVal(),f1rt->getError()),"");
      B0SigmaRT = sqrt(Sigma1RT*Sigma1RT*WG1+(1.-WG1)*Sigma2RT*Sigma2RT);
     }else{
      B0SigmaRT = Sigma1RT;
     }

      double B0sigma_wt =sigma_wt->getVal();
//      if(sigma_wt2){
//       double Sigma2WT =sigma_wt2->getVal();
//       double WG1=f3wt->getVal();
//       leg_sign->AddEntry(masHist ,Form( "f^{WT%d} =   %5.5f  #pm %5.5f",Q2Bin,f3wt->getVal(),f3wt->getError()),"");
//       B0sigma_wt = sqrt(Sigma1WT*Sigma1WT*WG1+(1.-WG1)*Sigma2WT*Sigma2WT);
//      }else{
//       B0sigma_wt = Sigma1WT;
//      }
     
     
     B0SigmaTemp = sqrt(B0SigmaRT*B0SigmaRT*fraction->getVal()+(1.-fraction->getVal())*B0sigma_wt*B0sigma_wt);
     

     std::cout<<Form("B0SigmaRT = %f B0sigma_wt = %f B0SigmaTot = %f",B0SigmaRT,B0sigma_wt,B0SigmaTemp) <<std::endl;

//     tagged_mass->setRange("SignLeft" ,mean_rt->getVal(),XMaxSign);
//     tagged_mass->setRange("SignRight",XMinSign,mean_rt->getVal());
//      tagged_mass->setRange(5.1,5.4);
//
     TCanvas* ccdf_signal = new TCanvas("ccdf_signal","cdf",200,10,1200,600);
//     ccdf_signal->Divide(2,2);
     TCanvas* ccdf_rt = new TCanvas("ccdf_rt","cdf RT",200,10,1200,600);
     ccdf_rt->Divide(2,2);
     
     TCanvas* ccdf_signal_zoom = new TCanvas("ccdf_signal_zoom","cdf",200,10,1200,600);
     ccdf_signal_zoom->Divide(2,2);
     TCanvas* ccdf_rt_zoom = new TCanvas("ccdf_rt_zoom","cdf RT",200,10,1200,600);
     ccdf_rt_zoom->Divide(2,2);
     TF1 *  Func_signal = c_signalFunction.asTF( RooArgList(*tagged_mass) );
     RooAbsPdf* signalCdf = (RooAbsPdf*)c_signalFunction.createCdf(*tagged_mass);
     TF1 *  Func_signalCdf= signalCdf->asTF( RooArgList(*tagged_mass) );
//     RooAbsPdf* theRTgaussCdf = (RooAbsPdf*)theRTgauss->createCdf(*tagged_mass);
//     TF1 *  Func_theRTgaussCdf= theRTgaussCdf->asTF( RooArgList(*tagged_mass) );
     std::cout<<"Estimate %% probability for Full/RT Signal:"<<std::endl;
     double integTails= 0;
//     double XLeftLim  = 0;
//     double XRightLim = 0;
//      double SigmaEstL = 0;
//      double SigmaEstR = 0;
     Func_signal->SetLineColor(kRed);
     Func_signal->SetLineWidth(1.0);
//      for(double iSigma=1;iSigma<5;iSigma++){
//      
//        integTails= TMath::Erfc(iSigma/sqrt(2));//  Erfc(x) = (2/sqrt(pi)) Integral(exp(-t^2))dt between x and infinity; t=x/sqrt(2)
//        XLeftLim  = Func_signalCdf->GetX(integTails/2.);
//        XRightLim = Func_signalCdf->GetX(1-integTails/2.);
//        SigmaEstL = (fabs(mean_rt->getVal()-XLeftLim))/iSigma;
//        SigmaEstR = (fabs(mean_rt->getVal()-XRightLim))/iSigma;
//        std::cout<<Form("Full signal %f%% [%iSigma gauss sigma] in the Range %f<mass<%f average sigmaL=%f sigmaR=%f",1-integTails,int(iSigma),XLeftLim,XRightLim,SigmaEstL,SigmaEstR)<<std::endl;
//        ccdf_signal->cd(iSigma);
//        Func_signal->SetRange(4.8,5.6);
//        Func_signal->SetMaximum(1.);
//        Func_signal->Draw();
//        TF1* Clone_signal =  (TF1*) Func_signal->Clone();
//        Clone_signal->SetRange(XLeftLim,XRightLim);
//        Clone_signal->SetFillColor(kBlue);
//        Clone_signal->SetFillStyle(3013);
//        Clone_signal->SetLineWidth(1.);
//        Clone_signal->Draw("SAME FC");
// //
//        ccdf_signal_zoom->cd(iSigma);
//        Func_signal->SetRange(4.8,5.6);
//        Func_signal->SetMaximum(0.2);
//        Func_signal->Draw();
//        Clone_signal =  (TF1*) Func_signal->Clone();
//        Clone_signal->SetRange(XLeftLim,XRightLim);
//        Clone_signal->Draw("SAME FC");
//        XLeftLim  = Func_theRTgaussCdf->GetX(integTails/2.);
//        XRightLim = Func_theRTgaussCdf->GetX(1-integTails/2.);
//        SigmaEstL = (fabs(mean_rt->getVal()-XLeftLim))/iSigma;
//        SigmaEstR = (fabs(mean_rt->getVal()-XRightLim))/iSigma;
//        std::cout<<Form("RT   signal %f%% [%iSigma gauss sigma] in the Range %f<mass<%f average sigmaL=%f sigmaR=%f",1-integTails,int(iSigma),XLeftLim,XRightLim,SigmaEstL,SigmaEstR)<<std::endl;
//
//        ccdf_rt->cd(iSigma);
//        Func_theRTgauss->SetRange(5.0,5.6);
//        Func_theRTgauss->SetMaximum(1.);
//        Func_theRTgauss->Draw();
//        TF1* Clone_theRTgauss =  (TF1*) Func_theRTgauss->Clone();
//        Clone_theRTgauss->SetRange(XLeftLim,XRightLim);
//        Clone_theRTgauss->SetFillColor(kBlue);
//        Clone_theRTgauss->SetFillStyle(3013);
//        Clone_theRTgauss->SetLineWidth(1.);
//        Clone_theRTgauss->Draw("SAME FC");
//        ccdf_rt->Update();
//        //
//        ccdf_rt_zoom->cd(iSigma);
//        Func_theRTgauss->SetRange(5.0,5.6);
//        Func_theRTgauss->SetMaximum(0.3);
//        Func_theRTgauss->Draw();
//        Clone_theRTgauss =  (TF1*) Func_theRTgauss->Clone();
//        Clone_theRTgauss->SetRange(XLeftLim,XRightLim);
//        Clone_theRTgauss->Draw("SAME FC");
//        ccdf_rt_zoom->Update();
//     }
//      ccdf_signal  ->Print(Form("cdf_signal_Q2Bin_%d.png",Q2Bin));
//      ccdf_rt->Print(Form("cdf_rt_Q2Bin_%d.png",Q2Bin));
//      ccdf_signal_zoom  ->Print(Form("cdf_signal_zoom_Q2Bin_%d.png",Q2Bin));
//      ccdf_rt_zoom->Print(Form("cdf_rt_zoom_Q2Bin_%d.png",Q2Bin));
//     tagged_mass->setRange("testRange",XLeftLim,XRightLim);
//     RooAbsReal*  testRTRange= theRTgauss->createIntegral(*tagged_mass,NormSet(*tagged_mass),Range("testRange"));
//     std::cout<<Form(" %f%% in the Range %f<mass<%f",testRTRange->getVal(),XLeftLim,XRightLim)<<std::endl;
//      TCanvas* ccdf = new TCanvas("ccdf","cdf",200,10,750,800);
//      ccdf->cd();
//      Func_signalCdf->Draw();
//      ccdf->Print("cdf.png");
     


//      tagged_mass->setRange("3sigmaintegral",mean_rt->getVal()-3*B0SigmaTemp,mean_rt->getVal()+3*B0SigmaTemp);
//      RooAbsReal* BckgInt3Sigma = bkg_exp->createIntegral(*tagged_mass,*tagged_mass,"3sigmaintegral");
//      RooAbsReal* SignInt3Sigma = c_signalFunction.createIntegral(*tagged_mass,*tagged_mass,"3sigmaintegral");
//      NBckgInt3Sigma = BckgInt3Sigma->getVal()*nbkg.getVal();
//      NSignInt3Sigma = SignInt3Sigma->getVal()*nsig.getVal();
//      std::cout<<Form("Bckg event +/- 3 sigma_w from mean RT = %f",NBckgInt3Sigma) <<std::endl;
//      std::cout<<Form("Sign event +/- 3 sigma_w from mean RT = %f",SignInt3Sigma) <<std::endl;
//      std::cout<<Form("Sign %% in +/- 3 sigma_w from mean RT = %f",SignInt3Sigma->getVal()) <<std::endl;
     integTails= TMath::Erfc(sqrt(2));
//      XLeftLim  = Func_theRTgaussCdf->GetX(integTails/2.);
//      XRightLim = Func_theRTgaussCdf->GetX(1-integTails/2.);
     XLeftSet  = Func_signalCdf->GetX(integTails/2.);
     XRightSet = Func_signalCdf->GetX(1-integTails/2.);
     tagged_mass->setRange("2sigmaRTintegral",XLeftSet,XRightSet);
     RooAbsReal* BckgInt2SigmaRT = bkg_exp->createIntegral(*tagged_mass,*tagged_mass,"2sigmaRTintegral");
     RooAbsReal* SignInt2SigmaRT = c_signalFunction.createIntegral(*tagged_mass,*tagged_mass,"2sigmaRTintegral");
     NBckgInt2Sigma = BckgInt2SigmaRT->getVal()*nbkg.getVal();
     NSignInt2Sigma = SignInt2SigmaRT->getVal()*nsig.getVal();
     std::cout<<Form("Bckg event %f<mass<%f from mean RT = %f",XLeftSet,XRightSet,NBckgInt2Sigma) <<std::endl;
     std::cout<<Form("Sign event %f<mass<%f from mean RT = %f",XLeftSet,XRightSet,NSignInt2Sigma) <<std::endl;
     std::cout<<Form("Sign %% in %f<mass<%f from mean RT = %f",XLeftSet,XRightSet,SignInt2SigmaRT->getVal()) <<std::endl;
//==============================================================
//===
//===   MODIFICA della scelta delle SideBand
//===
//==============================================================
//
     if(SigmaProbSign==0){
      std::cout<<Form("FitMassSpectrumRoofit::WARNING: Limits from sigma gauss model")<<std::endl;
      XMinSBL = B0Mass - NSigma1L*B0SigmaTemp;
      XMaxSBL = B0Mass - NSigma2L*B0SigmaTemp;
//      
      XMinSBR = B0Mass + NSigma1R*B0SigmaTemp;
      XMaxSBR = B0Mass + NSigma2R*B0SigmaTemp;
//
     }else if(SigmaProbSign==-1){
      std::cout<<Form("FitMassSpectrumRoofit::WARNING: setting bare limits")<<std::endl;
      XMinSBL = NSigma1L;
      XMaxSBL = NSigma2L;
//      
      XMinSBR = NSigma1R;
      XMaxSBR = NSigma2R;
     }else if(SigmaProbSign==1){
      std::cout<<Form("FitMassSpectrumRoofit::WARNING: Limits from gauss sign prob")<<std::endl;
      integTails= TMath::Erfc(NSigma2L/sqrt(2));
      XMaxSBL	= Func_signalCdf->GetX(integTails/2.);//  Erfc(x) = (2/sqrt(pi)) Integral(exp(-t^2))dt between x and infinity; t=x/sqrt(2)
//
      std::cout<<Form("Sideband Left  Internal Limit&Prob [%f - %f]",XMaxSBL, integTails/2.)<<std::endl;
      
//       integTails= TMath::Erfc(fabs(NSigma2L-NSigma1L)/sqrt(2));
//       XMinSBL	= XMaxSBL-(fabs(mean_rt->getVal()-Func_signalCdf->GetX(integTails/2.)));
      XMinSBL   = NSigma1L;
      integTails= TMath::Erfc(NSigma1R/sqrt(2));
      XMinSBR	= Func_signalCdf->GetX(1-integTails/2.);
      std::cout<<Form("Sideband Right Internal Limit&Prob [%f - %f]",XMinSBR, integTails/2.)<<std::endl;
//       integTails= TMath::Erfc(fabs(NSigma2R-NSigma1R)/sqrt(2));
//       XMaxSBR	= XMinSBR+(fabs(mean_rt->getVal()-Func_signalCdf->GetX(1-integTails/2.)));
      XMaxSBR   = NSigma2R;
     }else{
      std::cout<<Form("Sideband Definition=>>SigmaProbSign: 	INVALID OPTION: %d !!! Exit...",SigmaProbSign)<<std::endl;
      exit(1);
     } 
//     
     std::cout<<Form("Sideband Left  [%f-%f]",XMinSBL, XMaxSBL)<<std::endl;
     std::cout<<Form("Sideband Right [%f-%f]",XMinSBR, XMaxSBR)<<std::endl;

     ccdf_signal ->cd();
     Func_signal->SetRange(XMinSign,XMaxSign);
//     Func_signal->SetNormalized(true);
     TF1* Clone_signal  =  (TF1*) Func_signal->Clone();
     TF1* Clone_signalL =  (TF1*) Func_signal->Clone();
     TF1* Clone_signalR =  (TF1*) Func_signal->Clone();
//     Clone_signal->SetNormalized(true);
//     double Func_signal_Integ = Clone_signal->Integral(5.0,5.6);
//     std::cout<<"Func_signal_Integ ="<<Func_signal_Integ<<std::endl;
//     Clone_signal->GetHistogram()->Scale(1/Func_signal_Integ);
     Clone_signal->SetMaximum(0.1);
     Clone_signal->Draw();
//     Clone_signalL->SetNormalized(true);
//     double Func_signal_IntegL = Clone_signal->Integral(XMinSBL,XMaxSBL);
     Clone_signalL->SetRange(XMinSBL,XMaxSBL);
//     std::cout<<"Func_signal_IntegL ="<<Func_signal_IntegL<<std::endl;
//     Clone_signalL->GetHistogram()->Scale(Func_signal_IntegL/Func_signal_Integ);
     Clone_signalL->SetFillColor(kBlue);
     Clone_signalL->SetFillStyle(3013);
     Clone_signalL->SetLineWidth(1.);
     Clone_signalL->Draw("SAME FC");
//     Clone_signalR->SetNormalized(true);
//     double Func_signal_IntegR = Clone_signal->Integral(XMinSBR,XMaxSBR);
     Clone_signalR->SetRange(XMinSBR,XMaxSBR);
//     std::cout<<"Func_signal_IntegR ="<<Func_signal_IntegR<<std::endl;
//     Clone_signalR->GetHistogram()->Scale(Func_signal_IntegR/Func_signal_Integ);
     Clone_signalR->SetFillColor(kBlue);
     Clone_signalR->SetFillStyle(3013);
     Clone_signalR->SetLineWidth(1.);
     Clone_signalR->Draw("SAME FC");
     ccdf_signal ->Print(Form("cdf_signal_Q2Bin_%d.png",Q2Bin));
//     std::cout<<Form("Func_signal_IntegL%=%f Func_signal_IntegR%=%f",Func_signal_IntegL/Func_signal_Integ,Func_signal_IntegR/Func_signal_Integ)<<std::endl;
//exit(0);
//     RooRealVar* tagged_massS = new RooRealVar("tagged_mass" , "#mu^{+}#mu^{-}K#pi mass", 5.0, 5.6, "GeV");
//     RooRealVar* tagged_massF = new RooRealVar("tagged_mass" , "#mu^{+}#mu^{-}K#pi mass", 4.9, 5.6, "GeV");
//     tagged_mass->setRange(XMinSign,XMaxSign);
//     tagged_mass->setRange("full",XMinSign,XMaxSign);
     tagged_mass->setRange( "SBLeft"  ,XMinSBL,XMaxSBL);
     tagged_mass->setRange( "SBRight" ,XMinSBR,XMaxSBR);
//      tagged_massS->setRange("SBLeftS" ,XMinSBL,XMaxSBL);
//      tagged_massS->setRange("SBRightS",XMinSBR,XMaxSBR);
//      tagged_massF->setRange("SBLeftF" ,XMinSBL,XMaxSBL);
//      tagged_massF->setRange("SBRightF",XMinSBR,XMaxSBR);
     
     double tagged_mass_rangeFitMin=tagged_mass->getMin("full");
     double tagged_mass_rangeFitMax=tagged_mass->getMax("full");
     tagged_mass_rangeValMin=tagged_mass->getMin();
     tagged_mass_rangeValMax=tagged_mass->getMax();
     std::cout<<Form("%f<[tagged_mass fit mass range]<%f",tagged_mass_rangeFitMin,tagged_mass_rangeFitMax)<<std::endl; 
     std::cout<<Form("%f<[tagged_mass val mass range]<%f",tagged_mass_rangeValMin,tagged_mass_rangeValMax)<<std::endl; 
     
//     RooExponential* bkgEXP = new RooExponential(bkg_exp);
  
//      RooAbsReal* BckgAll = bkg_exp->createIntegral(*tagged_mass);
//      RooAbsReal* BckgFull = bkg_exp->createIntegral(*tagged_massF);
//      RooAbsReal* BckgFits = bkg_exp->createIntegral(*tagged_massS);
//      std::cout<<Form("Integrals All=%f Fits=%f and Full=%f",BckgAll->getVal(),BckgFits->getVal(),BckgFull->getVal())<<std::endl; 
     
     RooAbsReal* BckgSBL = bkg_exp->createIntegral(*tagged_mass,*tagged_mass,"SBLeft");
//     RooAbsReal* BckgSBL = bkg_exp->createIntegral(*tagged_mass,NormSet(*tagged_mass),Range("SBLeft"));
     RooAbsReal* BckgSBR = bkg_exp->createIntegral(*tagged_mass,*tagged_mass,"SBRight");
     double BckgEventsSBL = BckgSBL->getVal()*nbkg.getVal();
     double BckgEventsSBR = BckgSBR->getVal()*nbkg.getVal();
//     

     RooAbsReal* SignSBL = c_signalFunction.createIntegral(*tagged_mass,*tagged_mass,"SBLeft");
     RooAbsReal* SignSBR = c_signalFunction.createIntegral(*tagged_mass,*tagged_mass,"SBRight");
     double SignEventsSBL = SignSBL->getVal()*nsig.getVal();
     double SignEventsSBR = SignSBR->getVal()*nsig.getVal();
//     
     RooAbsReal* SignSBL_wt = c_WTgauss->createIntegral(*tagged_mass,*tagged_mass,"SBLeft");
     RooAbsReal* SignSBR_wt = c_WTgauss->createIntegral(*tagged_mass,*tagged_mass,"SBRight");
     double SignEventsSBL_wt = SignSBL_wt->getVal()*nsig.getVal()*(1-fraction->getVal());
     double SignEventsSBR_wt = SignSBR_wt->getVal()*nsig.getVal()*(1-fraction->getVal());
//     
     RooAbsReal* SignSBL_rt = c_RTgauss->createIntegral(*tagged_mass,*tagged_mass,"SBLeft");
     RooAbsReal* SignSBR_rt = c_RTgauss->createIntegral(*tagged_mass,*tagged_mass,"SBRight");
     double SignEventsSBL_rt = SignSBL_rt->getVal()*nsig.getVal()*fraction->getVal();
     double SignEventsSBR_rt = SignSBR_rt->getVal()*nsig.getVal()*fraction->getVal();
//     
     RooAbsReal* ModelSBL = fitFunction.createIntegral(*tagged_mass,*tagged_mass,"SBLeft");
     RooAbsReal* ModelSBR = fitFunction.createIntegral(*tagged_mass,*tagged_mass,"SBRight");
     double ModelEventsSBL = ModelSBL->getVal()*(nsig.getVal()+nbkg.getVal());
     double ModelEventsSBR = ModelSBR->getVal()*(nsig.getVal()+nbkg.getVal());
     double RealEventsSBL  = data->sumEntries(Form("tagged_mass>%f&&tagged_mass<%f",XMinSBL,XMaxSBL)) ;
     double RealEventsSBR  = data->sumEntries(Form("tagged_mass>%f&&tagged_mass<%f",XMinSBR,XMaxSBR)) ;
//
     RooAbsReal* SignalFull  = c_signalFunction.createIntegral(*tagged_mass,*tagged_mass,"full");
//     RooAbsReal* BckgFull    = bkg_exp->createIntegral(*tagged_mass,*tagged_mass,"full");
     RooAbsReal* ModelFull   = fitFunction.createIntegral(*tagged_mass,*tagged_mass,"full");

//     
     std::cout<<Form("Signal Norm Check = %f",SignalFull->getVal())  <<std::endl;
     std::cout<<Form("Estimated Bckg	events SB Left = %f",BckgEventsSBL)  <<std::endl;
     std::cout<<Form("Estimated Bckg	events SB Right= %f",BckgEventsSBR)  <<std::endl;
     std::cout<<Form("Estimated Bckg	events SB Total= %f",BckgEventsSBL+BckgEventsSBR)  <<std::endl;
     std::cout<<Form("========> real	events SB Left = %f",RealEventsSBL ) <<std::endl;
     std::cout<<Form("========> real	events SB Right= %f",RealEventsSBR ) <<std::endl;
     std::cout<<Form("========> real	events SB Total= %f",RealEventsSBL+RealEventsSBR ) <<std::endl;
     std::cout<<Form("========> real	all events inside [full] range Total= %f",data->sumEntries(Form("tagged_mass>%f&&tagged_mass<%f",tagged_mass_rangeMin,tagged_mass_rangeMax)) ) <<std::endl;
     std::cout<<Form("========> Fit	all events inside [full] range Total= %f",ModelFull->getVal()*(nsig.getVal()+nbkg.getVal())) <<std::endl;
     std::cout<<Form("========> real	all events Total= %f",data->sumEntries(Form("tagged_mass>%f&&tagged_mass<%f",XMinSign,XMaxSign)) ) <<std::endl;
     std::cout<<Form("========> Fit	all events Total= %f",(nsig.getVal()+nbkg.getVal())) <<std::endl;
//
     std::cout<<Form("Estimated Sign	events SB Left = %f",SignEventsSBL)  <<std::endl;
     std::cout<<Form("Estimated Sign	events SB Right= %f",SignEventsSBR)  <<std::endl;
     std::cout<<Form("Estimated Sign WT events SB Left = %f",SignEventsSBL_wt)  <<std::endl;
     std::cout<<Form("Estimated Sign WT	events SB Right= %f",SignEventsSBR_wt)  <<std::endl;
     std::cout<<Form("Estimated Sign RT events SB Left = %f",SignEventsSBL_rt)  <<std::endl;
     std::cout<<Form("Estimated Sign RT	events SB Right= %f",SignEventsSBR_rt)  <<std::endl;
     std::cout<<Form("Estimated Model	events SB Left = %f",ModelEventsSBL) <<std::endl;
     std::cout<<Form("Estimated Model	events SB Right= %f",ModelEventsSBR) <<std::endl;
     std::cout<<Form("events signal in SB [all]/ signal Tot  = %f",(SignEventsSBL+SignEventsSBR)/ nsig.getVal())<<std::endl;
     std::cout<<Form("events signal in SB [RT] / signal Tot  = %f",(SignEventsSBL_rt+SignEventsSBR_rt)/ nsig.getVal())<<std::endl;
     std::cout<<Form("events signal in SB [WT] / signal Tot  = %f",(SignEventsSBL_wt+SignEventsSBR_wt)/ nsig.getVal())<<std::endl;
     std::cout<<Form("events signal in SB [all]/ SB     Tot  = %f",(SignEventsSBL+SignEventsSBR)/ (RealEventsSBL+RealEventsSBR))<<std::endl;
     std::cout<<Form("events signal in SB [RT] / SB     Tot  = %f",(SignEventsSBL_rt+SignEventsSBR_rt)/ (RealEventsSBL+RealEventsSBR))<<std::endl;
     std::cout<<Form("events signal in SB [WT] / SB     Tot  = %f",(SignEventsSBL_wt+SignEventsSBR_wt)/ (RealEventsSBL+RealEventsSBR))<<std::endl;
     std::cout<<Form("events signal in SB Left / signal Tot  = %f",(SignEventsSBL)/ nsig.getVal())<<std::endl;
     std::cout<<Form("events signal in SB Right/ signal Tot  = %f",(SignEventsSBR)/ nsig.getVal())<<std::endl;
     std::cout<<Form("events signal in SB Left / SB     Tot  = %f",(SignEventsSBL)/ (RealEventsSBL+RealEventsSBR))<<std::endl;
     std::cout<<Form("events signal in SB Right/ SB     Tot  = %f",(SignEventsSBR)/ (RealEventsSBL+RealEventsSBR))<<std::endl;
     std::cout<<Form("events signal in SB Left / SB     Left = %f",(SignEventsSBL)/ (RealEventsSBL))<<std::endl;
     std::cout<<Form("events signal in SB Right/ SB     Right= %f",(SignEventsSBR)/ (RealEventsSBR))<<std::endl;
     std::cout<<"\n"<<std::endl;
     std::cout<<Form("%d &  %3.2f-%3.2f & %3.2f-%3.2f & %5.3f & %5.3f & %5.3f & %5.3f & %5.3f & %5.3f & %5.3f & %3.2f\\\\ \n",Q2Bin,XMinSBL,XMaxSBL,XMinSBR,XMaxSBR,\
     (SignEventsSBL)/nsig.getVal(),(SignEventsSBR)/nsig.getVal(), (SignEventsSBL)/ (RealEventsSBL),\
     (SignEventsSBR)/ (RealEventsSBR),\
     (SignEventsSBL_rt+SignEventsSBR_rt)/(RealEventsSBL+RealEventsSBR),\
     (SignEventsSBL_wt+SignEventsSBR_wt)/(RealEventsSBL+RealEventsSBR),\
     (SignEventsSBL+SignEventsSBR)/(RealEventsSBL+RealEventsSBR),\
     (RealEventsSBL+RealEventsSBR)/NBckgInt2Sigma) <<std::endl;
//     
     
     double xbinw = pdfHist->GetXaxis()->GetBinWidth(1);
     cout<<"Binw pdfHist ="<<xbinw<<endl;
//      for (int i = 1; pdfHist->GetNbinsX(); ++i) {
//         double xmass = xbinw/2.+(i-1)*xbinw;
// //      const RooArgSet * dataLoad = data->get (i);
// //      double xmass = dataLoad->getRealValue(tagged_mass->GetName());
// 	pdfHist->Fill(xmass, fitFunction.evaluate() );
// 	sigHist->Fill(xmass, c_signalFunction.evaluate());
// 	bkgHist->Fill(xmass, c_frt.evaluate());
//      }
     double NStepMass  = pdfHist->GetNbinsX();
     double NBINFactor = NStepMass/masHist->GetNbinsX();
     fitFunction.fillHistogram(pdfHist,*tagged_mass);
     c_signalFunction.fillHistogram(sigHist,*tagged_mass);
     bkg_exp->fillHistogram(bkgHist,*tagged_mass);
     tagged_mass->setRange(XMinSign,XMaxSign);
     RooAbsReal* BckgFullW   = bkg_exp->createIntegral(*tagged_mass,*tagged_mass);
     RooAbsReal* SignalFullW = c_signalFunction.createIntegral(*tagged_mass,*tagged_mass);
     RooAbsReal* ModelFullW  = fitFunction.createIntegral(*tagged_mass,*tagged_mass);
     
//     tagged_mass->setRange("full1",5.0,5.6);
     RooAbsReal* BckgFullS   = bkg_exp->createIntegral(*tagged_mass,*tagged_mass,"full");
     RooAbsReal* SignalFullS = c_signalFunction.createIntegral(*tagged_mass,*tagged_mass,"full");
     RooAbsReal* ModelFullS  = fitFunction.createIntegral(*tagged_mass,*tagged_mass,"full");
     
     double scaleModelW  = ModelFullW->getVal()/ModelFullS->getVal();
     double scaleSignalW = SignalFullW->getVal()/SignalFullS->getVal();
     double scaleBckgW   = BckgFullW->getVal()/BckgFullS->getVal();

     int MinBinRangeFit = pdfHist->GetXaxis()->FindBin(tagged_mass_rangeMin);
     int MaxBinRangeFit = pdfHist->GetXaxis()->FindBin(tagged_mass_rangeMax);
     std::cout<<Form("pdfHist (=model of mass spectrum) MinBinRangeFit = %d MaxBinRangeFit = %d NumBins = %d",MinBinRangeFit,MaxBinRangeFit,pdfHist->GetNbinsX()) <<std::endl;
//      pdfHist->Scale(NBINFactor*(nsig.getVal()+nbkg.getVal())*scaleModelW);
//      sigHist->Scale(NBINFactor*(nsig.getVal())*scaleSignalW);
//      bkgHist->Scale(NBINFactor*(nbkg.getVal())*scaleBckgW);
     pdfHist->Scale(NBINFactor*(nsig.getVal()+nbkg.getVal()));
     sigHist->Scale(NBINFactor*(nsig.getVal()));
     bkgHist->Scale(NBINFactor*(nbkg.getVal()));
//     cout<< Form("Integ Model (4.9-5.6) =%f  Integ Model (5.0-5.6) =%f scaleModelW=%f", ModelFullW->getVal(),ModelFullS->getVal(),scaleModelW)<<endl;
//     cout<< Form("Integ Signal(4.9-5.6) =%f  Integ Signal(5.0-5.6) =%f scaleSignalW=%f", SignalFullW->getVal(),SignalFullS->getVal(),scaleSignalW)<<endl;
//     cout<< Form("Integ Bckg  (4.9-5.6) =%f  Integ Bckg  (5.0-5.6) =%f scaleBckgW=%f", BckgFullW->getVal(),BckgFullS->getVal(),scaleBckgW)<<endl;
     cout<< Form("Integ Model (%f-%f) =%f  Integ Model (%f-%f) =%f scaleModelW=%f" ,XMinSign,XMaxSign,ModelFullW->getVal() ,tagged_mass_rangeMin,tagged_mass_rangeMax,ModelFullS->getVal() ,scaleModelW)<<endl;
     cout<< Form("Integ Signal(%f-%f) =%f  Integ Signal(%f-%f) =%f scaleSignalW=%f",XMinSign,XMaxSign,SignalFullW->getVal(),tagged_mass_rangeMin,tagged_mass_rangeMax,SignalFullS->getVal(),scaleSignalW)<<endl;
     cout<< Form("Integ Bckg  (%f-%f) =%f  Integ Bckg  (%f-%f) =%f scaleBckgW=%f"  ,XMinSign,XMaxSign,BckgFullW->getVal()  ,tagged_mass_rangeMin,tagged_mass_rangeMax,BckgFullS->getVal(),scaleBckgW)<<endl;
//exit(1);
   //   int MinBinSBL = bkgHist->GetXaxis()->FindBin(mean_rt->getVal()-3*B0SigmaTemp);
//      int MaxBinSBL = bkgHist->GetXaxis()->FindBin(mean_rt->getVal()+3*B0SigmaTemp);
//      double BckgIntSBL= nbkg.getVal()*bkgHist->Integral(MinBinSBL,MaxBinSBL)/bkgHist->Integral(MinBinRangeFit,MaxBinRangeFit);
//      std::cout<<Form("pdfHist (=model of mass spectrum) MinBinSBL = %d MaxBinSBL = %d Integ = %f",MinBinSBL,MaxBinSBL,BckgIntSBL) <<std::endl;
//  
     masHist->GetXaxis()->SetTitle("Mass (GeV/c^{2})");
     masHist->SetMarkerStyle(8);
     masHist->SetMarkerSize(MarkerSizeSet);
     masHist->SetTitle("");
     masHist->Draw("E1");
//     masHist.Draw("p");
     pdfHist->GetXaxis()->SetRangeUser(tagged_mass_rangeMin,tagged_mass_rangeMax);
     sigHist->GetXaxis()->SetRangeUser(tagged_mass_rangeMin,tagged_mass_rangeMax);
     bkgHist->GetXaxis()->SetRangeUser(tagged_mass_rangeMin,tagged_mass_rangeMax);
     pdfHist->SetLineWidth(PlotLineWidth);
     pdfHist->SetFillColor(0);
     pdfHist->SetLineColor(kBlue);
     pdfHist->Draw("same,HIST C");
     sigHist->SetLineWidth(PlotLineWidth);
     sigHist->SetLineColor(kMagenta);
     sigHist->SetLineStyle(kDashed);
     sigHist->SetFillColor(0);
     sigHist->Draw("same,HIST C");
     bkgHist->SetLineWidth(PlotLineWidth);
     bkgHist->SetLineColor(kRed);
     bkgHist->SetLineStyle(kDashed);
     bkgHist->SetFillColor(0);
     bkgHist->Draw("same,HIST C");
     leg_sign->Draw("same");
  
  fitMassFile->Close();
  return B0SigmaTemp;

}
//=========================================================================================

RooGaussian* _constrainVar(RooRealVar *var,RooWorkspace *w){
    
//    float constr[2] = *_getFittedVar(var.GetName(), w);
    RooRealVar c_val(Form("c_val_%s", var->GetName()),Form("c_val_%s", var->GetName()),var->getVal());
    RooRealVar c_err(Form("c_err_%s", var->GetName()),Form("c_err_%s", var->GetName()),var->getError());
    RooGaussian* gauss_constr =
                            new RooGaussian(   Form("c_%s", var->GetName()) , 
                                Form("c_%s", var->GetName()) , 
                                *var         ,  
                                RooConst( var->getVal() ), 
                                RooConst( var->getError() ) 
                                ) ;
    std::cout<< Form("constraining var %s: %f with uncertainty %f - limits [%f , %f]",var->GetName(),c_val.getVal(),c_err.getVal(),var->getMin(),var->getMax())<<std::endl;  
    if(Q2Bin==3&&RunEra==2016){
    double checkMax = var->getVal()+ 7*var->getError();			      
    double checkMin = var->getVal()- 7*var->getError();			      
    std::cout<< Form("Warning in _constrainVar: limits for %s, from [%f,%f] ==> [%f,%f]\n",var->GetName(),var->getMin(),var->getMax(),checkMin,checkMax);
    var->setMax(checkMax) ;	     
    var->setMin(checkMin) ;	     
    std::cout<< Form("Warning: redifine limits var %s: %f with uncertainty %f - limits [%f , %f]",var->GetName(),c_val.getVal(),c_err.getVal(),var->getMin(),var->getMax())<<std::endl;  
    }                        
    return gauss_constr;
}                           
//=========================================================================================
//
//=========================================================================================
// float*  _getFittedVar(const char* varName,RooWorkspace w=0){
//     float out[2];
//     if (&w!=0){
//         out[0]=w->var(varName).getVal();
// 	out[1]=w->var(varName).getError();
//     }else{
//         out[0]=varName.getVal();
// 	out[1]=varName.getError();
//     }
//     return out;	
// }
// double FitMassSpectrumRoofit(UnbinnedDataSet* dataMass, TCanvas* c2, TH1D* masHist, TH1D* pdfHist, TH1D*sigHist, TH1D* bkgHist, int MaxDegreeBckg){
// //
//   if(MaxDegreeBckg<=0) {
//    cout<<"**********************************************************************\n"<<endl;
//    cout<<"Error!! MaxDegree <=0 in   FitMassSpectrumRoofit		       *\n"<<endl;
//    cout<<"**********************************************************************\n"<<endl;
//   }
// //
//   double B0SigmaTemp = 0.
//   double xmeanRTSign = 5.280;
//   RooFit::RooRealVar meanRTSign("meanRTSign"  ,xmeanRTSign,XStepMinuit, 5., 5.5);
//   RooFit::RooRealVar mean_wtSign("mean_wtSign"  ,xmeanRTSign,XStepMinuit, 5., 5.5);
//   RooFit::RooRealVar sigmaRTSign1("sigmaRTSign1",0.0139,XStepMinuit, 0., 1.);
//   RooFit::RooRealVar sigmaRTSign2("sigmaRTSign2",0.0228,XStepMinuit, 0., 1.);
//   RooFit::RooRealVar sigmaRTSign3("sigmaRTSign3",0.0601,XStepMinuit, 0., 1.);
//   RooFit::RooRealVar sigma_wtSignCB1("sigma_wtSignCB1",0.0139,XStepMinuit, 0., 1.);
//   RooFit::RooRealVar sigma_wtSignCB2("sigma_wtSignCB2",0.0228,XStepMinuit, 0., 1.);
//   RooFit::RooRealVar alpha1("alpha1",0.0139,XStepMinuit, 0., 1.);
//   RooFit::RooRealVar alpha2("alpha2",0.0228,XStepMinuit, 0., 1.);
//   RooFit::RooRealVar RTwg1("RTwg1",0.44, 0., 1.);
//   RooFit::RooRealVar RTwg2("RTwg2",0.5 , 0., 1.);
//   RooFit::RooRealVar WTwg1("WTwg1",0.44, 0., 1.);
//   RooFit::RooRealVar WTwg2("WTwg2",0.5 , 0., 1.);
// //
//   RooGaussian *gaussRTSign1 = new RooGaussian("gaussRTSign1","gauss Right Tagged 1"    ,xMassRoofit,meanRTSign,sigmaRTSign1) ;
//   RooGaussian *gaussRTSign2 = new RooGaussian("gaussRTSign2","gauss Right Tagged 2"    ,xMassRoofit,meanRTSign,sigmaRTSign2) ;
//   RooGaussian *gaussRTSign3 = new RooGaussian("gaussRTSign3","gauss Right Tagged 3"    ,xMassRoofit,meanRTSign,sigmaRTSign3) ;
// //  
//   
//   RooGaussian *cstr_sigmaRTSign1 = new RooGaussian("cstr_sigmaRTSign1","sigma1 MC constr." ,xMassRoofit,sigmaRTSign1MC,sigmaRTSign1MC_err) ;
//   RooGaussian *cstr_sigmaRTSign2 = new RooGaussian("cstr_sigmaRTSign2","sigma2 MC constr." ,xMassRoofit,sigmaRTSign2MC,sigmaRTSign2MC_err) ;
//   RooGaussian *cstr_sigmaRTSign3 = new RooGaussian("cstr_sigmaRTSign3","sigma3 MC constr." ,xMassRoofit,sigmaRTSign3MC,sigmaRTSign3MC_err) ;
// //
// //
//   RooGaussian *cstr_meanRTSign1  = new RooGaussian("cstr_meanRTSign1" , "mean1 MC constr." ,xMassRoofit,meanRTSign1MC,meanRTSign1MC_err) ;
//  
//   RooCBShape *crystalBallWTSign1 = new RooCBShape("crystalBallWTSign1", "crystalBall Wrong Tagged Sign1",xMassRoofit,mean_wtSign,sigma_wtSign1,alpha1,n1);
//   RooCBShape *crystalBallWTSign2 = new RooCBShape("crystalBallWTSign2", "crystalBall Wrong Tagged Sign2",xMassRoofit,mean_wtSign,sigma_wtSign2,alpha2,n2);
//   RooGaussian *gaussWTSign = new RooGaussian("gaussWTSign","gauss Right Tagged"    ,xMassRoofit,mean_wtSign,sigma_wtSign) ;
//   
// // Signal Mass RT  
//   RooAddPdf * RTSignalMass = 0;
//   if     (numRTGau=1){
//    RTSignalMass = &gaussRTSign1;
//   else if(numRTGau=2){
//    RTSignalMass = new RooAddPdf("RTSignalMass","RTSignalMass",RooArgList(*gaussRTSign1,*gaussRTSign1),RooArgList(RTwg1));
//   else if(numRTGau=3){
//    RTSignalMass = new RooAddPdf("RTSignalMass","RTSignalMass",RooArgList(*gaussRTSign1,*gaussRTSign2,*gaussRTSign3),RooArgList(RTwg1,RTwg2));
//   } 
// //
//   RooAddPdf * WTSignalMass = 0;
//   if     (numWTMod=1){
//    WTSignalMass = &gaussWTSign1;
//   else if(numWTMod=2){
//    WTSignalMass = new RooAddPdf("WTSignalMass","WTSignalMass",RooArgList(*gaussWTSign,crystalBallWTSign1),RooArgList(WTwg1));
//   else if(numWTMod=3){
//    WTSignalMass = new RooAddPdf("WTSignalMass","WTSignalMass",RooArgList(*gaussWTSign1,*gaussWTSign2,*gaussWTSign3),RooArgList(WTwg1,WTwg2));
//   } 
// //
//    slope	 = RooRealVar	 ("slope"      , "slope"	   ,	0.5,   -10, 10);
//    bkg_exp	 = RooExponential("bkg_exp"    , "exponential"     ,  slope,   tagged_mass  );
//    pol_c1	 = RooRealVar	 ("p1"         , "coeff x^0 term"  ,	0.5,   -10, 10);
//    pol_c2	 = RooRealVar	 ("p2"         , "coeff x^1 term"  ,	0.5,   -10, 10);
//    bkg_pol	 = RooChebychev  ("bkg_pol"    , "2nd order pol"   ,  tagged_mass, RooArgList(pol_c1,pol_c2));
//   
//    nsig 	 = RooRealVar("Yield"	      , "signal frac"	 ,    4000,	0,   1000000);
//    nbkg 	 = RooRealVar("nbkg"	      , "bkg fraction"   ,    1000,	0,   550000);
// 
//    return B0SigmaTemp;
// //
// }


double FitMassSpectrum(UnbinnedDataSet* dataMass, TCanvas* c2, TH1D* masHist, TH1D* pdfHist, TH1D*sigHist, TH1D* bkgHist, int MaxDegreeBckg){

   ///////////////////////////////////////////////////////
   //*****************************************************
   //*
   //*
   //*  	    FIT Mass Spectrum
   //*   
   //*
   //*****************************************************
   ///////////////////////////////////////////////////////
   
     if(MaxDegreeBckg<=0) {
      cout<<"**********************************************************************\n"<<endl;
      cout<<"Error!! MaxDegree <=0 in   FitMassSpectrum                           *\n"<<endl;
      cout<<"**********************************************************************\n"<<endl;
     }
     GooFit::Variable mean  ("mean"  ,5.2762,XStepMinuit, 5., 5.5);
     GooFit::Variable sigma1("sigma1",0.0139,XStepMinuit, 0., 1.);
     GooFit::Variable sigma2("sigma2",0.0228,XStepMinuit, 0., 1.);
     GooFit::Variable sigma3("sigma3",0.0601,XStepMinuit, 0., 1.);
     GooFit::Variable wg1("wg1",0.44, 0., 1.);
     GooFit::Variable wg2("wg2",0.5 , 0., 1.);

     

     GooFit::RGaussianPdf* signalMass1 = new GooFit::RGaussianPdf("signalMass1", xMass, mean, sigma1);
     GooFit::RGaussianPdf* signalMass2 = new GooFit::RGaussianPdf("signalMass2", xMass, mean, sigma2);

     std::vector<GooFit::Variable> weightsSignMass;
     weightsSignMass.push_back(wg1);
     std::vector<PdfBase*> compsSignMass;
     compsSignMass.push_back(signalMass1);
     compsSignMass.push_back(signalMass2);
     GooFit::AddPdf* signalMass= new AddPdf("signalMass", weightsSignMass, compsSignMass);
     GooFit::Variable* signalYield = new GooFit::Variable("signalYield",40000*NFactGen,	     0.,10000000.);
     GooFit::Variable* bckgYield   = new GooFit::Variable("bckgYield"  ,60000*NFactGen ,      0.,20000000.);

     
     GooFit::Variable p0("p0",-3.08433e-01,-1.,1. ); 
     GooFit::Variable p1("p1"	       ,0.,-1.,1. ); 
     GooFit::Variable VMinSign("VMinSign",XMinSign ); 
     GooFit::Variable VMaxSign("VMaxSign",XMaxSign ); 
     SimpleCheby2Pdf* SimpleCheby2  = new SimpleCheby2Pdf("SimpleCheby2", xMass, p0, p1,VMinSign,VMaxSign);
     GooFit::Variable ps0("ps0",1.58302e+01, 8.5 , 30.); 
     GooFit::Variable ps1("ps1",5.11588e+00,   4.0,5.2);
     GooFit::Variable ps2("ps2",1.); 
     GooFit::Variable ps3("ps3",0.);
     ErfcMassPdf* ErfcMassBckg = new ErfcMassPdf("ErfcMassBckg",xMass,ps0,ps1,ps2,ps3);;
     
      GooFit::Variable        wb1("wb1",0.3, 0., 1.);
      std::vector<GooFit::Variable> weightsBckgMass;
      weightsBckgMass.push_back(wb1);
   //	weightsBckgMass.push_back(wb2);
   // //  weightsBckgMass.push_back(wb3);
   // 
   // //  ArgusPdf* argus = new  ArgusPdf("argus", xMass, treshold, aslope, true, apower);  
   // 
      std::vector<PdfBase*> compsBckgMass;
   // //  compsBckgMass.push_back(gaussBckgB0);
   // //  compsBckgMass.push_back(gaussBckgB0);
   // //    compsBckgMass.push_back(argus);
   // //  compsBckgMass.push_back(poly);
      compsBckgMass.push_back(SimpleCheby2);
      compsBckgMass.push_back(ErfcMassBckg);
   //  
   //	GooFit::AddPdf* bckgMass= new AddPdf("bckgMass", weightsBckgMass, compsBckgMass);
   //
     GooFit::Variable b0("b0",	1.      );
     GooFit::Variable b1("b1",	0.     ,0.,1000. );
     GooFit::Variable b2("b2",	0.     ,0.,1000. );
     GooFit::Variable b3("b3",	0.     ,0.,1000. );
     GooFit::Variable b4("b4",	0.      );
     GooFit::Variable b5("b5",	0.     ,0.,1000. );
//      GooFit::Variable b6("b6"	       ,0.,0.,100. ); 
//      GooFit::Variable b7("b7"	       ,0.,0.,100. ); 

     std::vector<GooFit::Variable> ParBernBckg;
     ParBernBckg.push_back(b0);
     if(MaxDegreeBckg>=1) ParBernBckg.push_back(b1);
     if(MaxDegreeBckg>=2) ParBernBckg.push_back(b2);
     if(MaxDegreeBckg>=3) ParBernBckg.push_back(b3);
     if(MaxDegreeBckg>=4) ParBernBckg.push_back(b4);
     if(MaxDegreeBckg>=5) ParBernBckg.push_back(b5);
     std::vector<GooFit::Variable> Limits1D;
     Limits1D.push_back(VMinSign);
     Limits1D.push_back(VMaxSign);
   //
     FastBernsteinPdf* bckgMass  = new FastBernsteinPdf("bckgMass", xMass, ParBernBckg, Limits1D,MaxDegreeBckg);
     
     std::vector<PdfBase*> compsMass;
     compsMass.push_back(signalMass);
     compsMass.push_back(bckgMass);

     std::vector<GooFit::Variable> weightsYield;
     weightsYield.push_back(*signalYield);
     weightsYield.push_back(*bckgYield);

     GooFit::AddPdf modelMass("modelMass", weightsYield, compsMass); 
     modelMass.setData(dataMass);
     
     

   //  GooFit::FitManager fitter(&model);//
     int NumCalls = 12000;

     if(SetMinuit2){
      GooFit::FitManagerMinuit2 fitter(&modelMass);
      fitter.setMaxCalls(NumCalls);
      fitter.setVerbosity(2);
      fitter.fit();
     }else{
      std::cout<<"Warning !!!! bSetting num call for MINUIT :"<<NumCalls  <<std::endl;
      GooFit::FitManagerMinuit1 fitter(&modelMass);
      fitter.setMaxCalls(NumCalls);
      fitter.useHesseBefore(false);
      fitter.useHesse(boolHesse);
      fitter.useMinos(false);
      cout<<"\n"<<endl;
      cout<<"		       ===*** Start Fit ***=== "<<endl;
      cout<<"		       ===*** Start Fit ***=== "<<endl;
      cout<<"		       ===*** Start Fit ***=== "<<endl;
      cout<<"\n"<<endl;

      Minuit1 * Minuit = fitter.getMinuitObject();
      Minuit->SetPrintLevel(FitPrintLevel);
      fitter.fit();
     }

     UnbinnedDataSet gridMass(xMass);
     double totalDataMass = 0; 
//      double NStepMass = XHScale * xMassHBin;
//      double NBINFactor = NStepMass/xMassHBin2;
     double NStepMass  = pdfHist->GetNbinsX();
     double NBINFactor = NStepMass/masHist->GetNbinsX();
     for (int i = 0; i < NStepMass; ++i) {
       double step = (xMass.getUpperLimit() - xMass.getLowerLimit())/NStepMass;
       xMass.setValue(xMass.getLowerLimit() + (i + 0.5) * step);
       gridMass.addEvent(); 
       totalDataMass++; 
     }

     modelMass.setData(&gridMass);
     std::vector<std::vector<double> > pdfValsMass = modelMass.getCompProbsAtDataPoints();
   //  modelMass.getCompProbsAtDataPoints(pdfValsMass); 
     double totalPdfMass = 0; 
     double totalSigMass = 0; 
     double totalBkgMass = 0; 
     for (int i = 0; i < gridMass.getNumEvents(); ++i) {
       gridMass.loadEvent(i); 
       totalPdfMass += pdfValsMass[0][i]; 
       totalSigMass += pdfValsMass[1][i]; 
       totalBkgMass += pdfValsMass[2][i]; 
     }
     yieldSignal = signalYield->getValue();
     yieldBckg   = bckgYield->getValue();
     double yieldModel  = yieldSignal+yieldBckg;
     for (int i = 0; i < gridMass.getNumEvents(); ++i) {
       gridMass.loadEvent(i); 
       pdfHist->Fill(xMass.getValue(), NBINFactor*yieldModel*pdfValsMass[0][i]/totalPdfMass);
       sigHist->Fill(xMass.getValue(), NBINFactor*yieldSignal*pdfValsMass[1][i]/totalSigMass);
       bkgHist->Fill(xMass.getValue(), NBINFactor*yieldBckg*pdfValsMass[2][i]/totalBkgMass);
     }
     
     std::cout<<"totalPdfMass = "<< totalPdfMass<<std::endl;
     std::cout<<"Signal Yield = "<< yieldSignal<<std::endl;
     std::cout<<"Bckg	Yield = "<< yieldBckg<<std::endl;
     std::cout<<"Tot   Yield  = "<< yieldModel<<std::endl;
     
     double B0SigmaTemp =0.;
     double Sigma1 = sigma1.getValue();
     double Sigma2 = sigma2.getValue();
     double WG1    = wg1.getValue();
 
     c2->cd();
     TLegend* leg_sign = new TLegend(0.30,0.70,0.90,0.90);
     leg_sign->SetTextSize(0.025) ;
     leg_sign->SetTextAlign(31);
     leg_sign->SetBorderSize(0.);
     leg_sign->SetFillStyle(0);
     leg_sign->SetHeader("B^{0} mass spectrum  Fit Projection");
     if(signalYield->getError()!=0){
       leg_sign->AddEntry(masHist ,Form( "Yield_{Sign} =     %5.0f  #pm %5.0f",signalYield->getValue(),signalYield->getError()),"");
     }else{
       leg_sign->AddEntry(masHist ,Form( "Yield_{Sign} =     %5.0f Fixed",signalYield->getValue()),"");
     }
     if(bckgYield->getError()!=0){
       leg_sign->AddEntry(masHist ,Form( "Yield_{Bckg} =     %5.0f  #pm  %5.0f",bckgYield->getValue(),bckgYield->getError()),"");
     }else{
       leg_sign->AddEntry(masHist ,Form( "Yield_{Bckg} =     %5.0f  Fixed",bckgYield->getValue()),"");
     }
 
     if(mean.getError()!=0){
      leg_sign->AddEntry(masHist ,Form( "M_{B^{0}} =   %5.5f  #pm %5.5f",mean.getValue(),mean.getError()),"");
     }else{
      leg_sign->AddEntry(masHist ,Form( "M_{B^{0}} =   %5.5f Fixed",mean.getValue()),"");
      }
     if(sigma1.getError()!=XStepMinuit){
      leg_sign->AddEntry(masHist ,Form( "#sigma#scale[0.6]{1}_{B^{0}} =   %5.5f  #pm %5.5f",sigma1.getValue(),sigma1.getError()),"");
     }else{
      leg_sign->AddEntry(masHist ,Form( "#sigma#scale[0.6]{1}_{B^{0}} =   %5.5f Fixed",sigma1.getValue()),"");
     }
     if(sigma2.getError()!=XStepMinuit){
      leg_sign->AddEntry(masHist ,Form( "#sigma#scale[0.6]{2}_{B^{0}} =   %5.5f  #pm %5.5f",sigma2.getValue(),sigma2.getError()),"");
//   	}else{
//   	 leg_sign->AddEntry(masHist ,Form( "#sigma#scale[0.6]{2}_{B^{0}} =   %5.5f Fixed",sigma2.getValue()),"");
     }
     if(wg1.getError()!=XStepMinuit){
      leg_sign->AddEntry(masHist ,Form( "W_{Gaus} =   %5.5f  #pm %5.5f",wg1.getValue(),wg1.getError()),"");
      B0SigmaTemp = sqrt(Sigma1*Sigma1*WG1+(1.-WG1)*Sigma2*Sigma2);
     }else{
      B0SigmaTemp = Sigma1;
     }
     masHist->GetXaxis()->SetTitle("Mass (GeV/c^{2})");
     masHist->SetMarkerStyle(8);
     masHist->SetMarkerSize(MarkerSizeSet);
     masHist->SetTitle("");
     masHist->Draw("E1");
//     masHist.Draw("p");
     pdfHist->SetLineWidth(PlotLineWidth);
     pdfHist->SetFillColor(0);
     pdfHist->SetLineColor(kBlue);
     pdfHist->Draw("same,HIST C");
     sigHist->SetLineWidth(PlotLineWidth);
     sigHist->SetLineColor(kMagenta);
     sigHist->SetLineStyle(kDashed);
     sigHist->SetFillColor(0);
     sigHist->Draw("same,HIST C");
     bkgHist->SetLineWidth(PlotLineWidth);
     bkgHist->SetLineColor(kRed);
     bkgHist->SetLineStyle(kDashed);
     bkgHist->SetFillColor(0);
     bkgHist->Draw("same,HIST C");
     leg_sign->Draw("same");
  
  
  return B0SigmaTemp;

///////////////////////////////////////////////////////
//*****************************************************
//
//  END FIT Mass Spectrum
//
//*****************************************************
///////////////////////////////////////////////////////
  
}
//
//=========================================================================================
//
// Namelist Routine
//
std::map<std::string, std::string> ReadNamelist(int argc, char** argv){
   if ( argc>=1 && (strcmp(argv[0],"namelist")>=0) ){
     std::cout<<"Defined namelist: "<<argv[0]<<std::endl;
   }else{
     std::cout<<"Namelist:"<<argv[0]<<"  should be named/renamed namelist*.list "<<argc<<std::endl;
     exit(1);
   }
   std::vector<std::string> split( char *str, char c = ' ');
   ifstream indata;
   std::map<std::string, std::string> mappa;
   std::string line;
   std::vector<std::string>vstring ;
//
    indata.open(argv[0]);
   if(!indata) { // file couldn't be opened
   	std::cout <<"Line: "<<__LINE__ <<" "<<argv[0]<< " Error: fileList can not be opened" << std::endl;
   	exit(1);
   }
   while(std::getline(indata, line)) {
	 line.erase(std::remove(line.begin(), line.end(), '\t'), line.end());
	 line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
	 line.erase(std::remove(line.begin(), line.end(), ' ' ), line.end());

 	 char *cstr = new char [line.size()+1];


 	 strcpy (cstr, line.c_str());
//	 cout <<"stringa->"<< cstr << endl;
	 vstring = split(cstr,'=');
	 mappa.insert( std::pair<string,string>(vstring[0],vstring[1]) );
    }
    std::cout<<"//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////"<<std::endl;
    for (map<string,string>::iterator imap = mappa.begin();
    			       imap != mappa.end();
    			       ++imap)
    {
   	std::cout <<"mappa->"<< (*imap).first<<" = "<<(*imap).second << std::endl;
    }
    std::cout<<"//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////"<<std::endl;
    indata.close();	
  return mappa ;
}
//===============================================================================================================
std::vector<std::string> split( char *str, char c = ' ')
{
    std::vector<std::string> result;

    while(1)
    {
         char *begin = str;

        while(*str != c && *str)
                str++;

        result.push_back(string(begin, str));

        if(0 == *str++)
                break;
    }

    return result;
}
//===============================================================================================================
void replaceAll(std::string& str, const std::string& from, const std::string& to) {
    if(from.empty())
        return;
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
    }
}
//===============================================================================================================
void replaceChar(char * txt,const  char * txt1,const  char * txt2) {

  std::stringstream sss,sss1,sss2;
  sss<<txt;
  sss1<<txt1;
  sss2<<txt2;
  std::string ss=sss.str();
  replaceAll( ss,  sss1.str(), sss2.str());
  strcpy(txt,ss.c_str());
  sss.str("");
  sss.clear();
  sss1.str("");
  sss1.clear();
  sss2.str("");
  sss2.clear();
  printf ("replaceChar output=>%s\n",txt);
}  


