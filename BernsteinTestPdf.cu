#define MAX_THREADS_PER_BLOCK 512
#define MIN_BLOCKS_PER_MP     20
#include <goofit/PDFs/mypdf/BernsteinTestPdf.h>
#include <goofit/Variable.h>

//   __global__ void
//   __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)

namespace GooFit {

/* a struct for storing double numbers */
// struct bernVal {
//   double bernFunc;
//   double bernIntg;
// };


 __device__ __thrust_forceinline__ fptype device_coeffbinomial_ber(fptype enne, fptype kappa){
 
        fptype factor=1.;
        for(fptype i = 1; i <=kappa; ++i) {
          factor *= (enne+1-i)/i; 
        }	 
 
        if (factor<=0 ){
	 printf("Error in BernsteinTestPdf coeffbinomial=> factor = %f enne=%f kappa=%f",factor,enne,kappa);
         return 0;
	} 
       return factor;
}
 __device__ __thrust_forceinline__ fptype  device_BernsteinTestkn_func(fptype x, fptype enne, fptype kappa){
 
   return device_coeffbinomial_ber(enne,kappa)*pow(x,kappa)*pow(1.0-x,enne-kappa);


}
 __device__ fptype  device_BernsteinTestkn_intg(fptype x, fptype enne, fptype kappa){
 
//  	if ((52 == THREADIDX) && (0 == BLOCKIDX)){
//       printf("==================================================\n");
//       printf("==================================================\n");
//      }
//       struct bernVal results;
//       results.bernFunc = 0;
//       results.bernIntg = 0;
      if (x<0 || x>1 ){
       printf(" Error in BernsteinTestkn_intg  x=%5.15f out of range [0,1]\n",x);
       return 0.;
      }
//      if (kappa>enne) return 0;
//      bernkn *= pow(x,kappa) ;
//      bernkn *= pow(1.0-x,enne-kappa) ;
      fptype integbernkn = 0;
      fptype ifactni = 0;
      fptype ifactik = 0;
      
       for(fptype i = kappa; i <=enne ; ++i) {
// n!/(i!(n-i)!)
//         ifactni=1;
//         for(float l = 1; l <=i; ++l) {
//           ifactni *= (maxDegree+1-l)/l; 
//         }	 

        ifactni =  device_coeffbinomial_ber(enne,i);
// i!/(k!(i-k)!)
//         ifactik=1;
//         for(float l = 1; l <=k; ++l) {
//           ifactik *= (i+1-l)/l; 
//         }	
 
        ifactik =  device_coeffbinomial_ber(i,kappa);
//
//        bernkn      += ifactni*ifactik*pow(menuno,i-kappa)*pow(x, i) ;
        integbernkn += ifactni*ifactik*pow(-1.0,i-kappa)/(i+1);
//	if ((52 == THREADIDX) && (0 == BLOCKIDX)){
//          printf("pow(x=%5.15f,i=%5.15f)=%5.15f\n",x,i,pow(x, i ));
//          printf("pow(-1=%5.15f,i-kappa)=%5.15f\n",menuno,pow(menuno,i-kappa));
//          printf("BernsteinTestkn=%5.15f integral=%5.15f ifactni=%5.15f ifactik=%5.15f \n",bernkn,integbernkn,ifactni,ifactik);
//          printf("BernsteinTestkn=%f integral=%f kappa=%f i=%f enne=%f ni=%f nk=%f\n",bernkn,integbernkn,kappa,i,enne,ifactni,ifactik);
//        }
       }

       if (integbernkn<=0 ){
//	if ((52 == THREADIDX) && (0 == BLOCKIDX)){
         printf(" Error in BernsteinTestkn_intg x=%5.15f integral = %5.15f THREADIDX=%d BLOCKIDX=%d\n", x,kappa,enne,integbernkn,THREADIDX,BLOCKIDX);
//	}
       }
//       results.bernFunc = bernkn;
//       results.bernIntg = integbernkn;
       return integbernkn;
}
//=======================================================================================================================
 __device__ fptype  device_BernsteinTestkn_intgBin( fptype xLeft, fptype xRight, fptype enne, fptype kappa){
 
//  	if ((52 == THREADIDX) && (0 == BLOCKIDX)){
//       printf("==================================================\n");
//       printf("==================================================\n");
//      }
      fptype integbernkn = 0.0;
      fptype ifactni = 0.0;
      fptype ifactik = 0.0;
      fptype powxL = pow(xLeft ,kappa+1) ;
      fptype powxR = pow(xRight,kappa+1) ;
      
//	if( THREADIDX==98 && BLOCKIDX==0){ printf("=========================\n");}
       for(fptype i = kappa; i <=enne ; ++i) {
// n!/(i!(n-i)!)

        ifactni =  device_coeffbinomial_ber(enne,i);
// i!/(k!(i-k)!)
 
        ifactik =  device_coeffbinomial_ber(i,kappa);
//
//        bernkn      += ifactni*ifactik*pow(menuno,i-kappa)*pow(x, i) ;
//        integbernkn += ifactni*ifactik*pow(-1.0,i-kappa)*(pow(xRight, i+1)-pow(xLeft,i+1))/(i+1);
         integbernkn += ifactni*ifactik*pow(-1.0,i-kappa)*(powxR-powxL)/(i+1);
	 powxL*=xLeft;
	 powxR*=xRight;
// 	if( THREADIDX==98 && BLOCKIDX==0){
// 	 printf("integbernkn loop => enne=%15.15f kappa=%15.15f integbernkn+ = %15.15f ifactni=%15.15f ifactnk=%15.15f i=%15.15f\n",enne,kappa,ifactni*ifactik*pow(-1.0,i-kappa)/(i+1),ifactni,ifactik,i);
//   	}
//	if ((52 == THREADIDX) && (0 == BLOCKIDX)){
//          printf("pow(x=%5.15f,i=%5.15f)=%5.15f\n",x,i,pow(x, i ));
//          printf("pow(-1=%5.15f,i-kappa)=%5.15f\n",menuno,pow(menuno,i-kappa));
//          printf("EffiBernsteinTestkn=%5.15f integral=%5.15f ifactni=%5.15f ifactik=%5.15f \n",bernkn,integbernkn,ifactni,ifactik);
//          printf("EffiBernsteinTestkn=%f integral=%f kappa=%f i=%f enne=%f ni=%f nk=%f\n",bernkn,integbernkn,kappa,i,enne,ifactni,ifactik);
//        }
       }

       if (integbernkn<=0.0 ){
	if ((476 == THREADIDX) && (148 == BLOCKIDX)){
         printf(" Error in BernsteinTestkn_intgbin xLeft=%f xRight=%f kappa=%f enne=%f integral = %5.15f THREADIDX=%d BLOCKIDX=%d\n",xLeft,xRight,kappa,enne,integbernkn,THREADIDX,BLOCKIDX);
	}
        integbernkn=1.E-30;
       }
//       results.bernFunc = bernkn;
//       results.bernIntg = integbernkn;
// // 	if( THREADIDX==98 && BLOCKIDX==0){
// // 	 printf("integbernkn => enne=%15.15f kappa=%15.15f integbernkn = %15.15f \n",enne,kappa,integbernkn);
// //          printf("=========================\n");
// // 	}
       return integbernkn;
}
//
//
//
//
//=================================================================================================================================== 
//
//================ device_BernsteinTest ======================================
// 
//=================================================================================================================================== 
__device__ fptype device_BernsteinTest(fptype *evt, fptype *p, unsigned int *indices) {
    // Structure is nP lowestdegree c1 c2 c3 nO o1
     
     
//    struct bernVal bernknval;

    int numParams = (indices[0]) ;
    int maxDegree = (indices[1]);

    fptype x   = evt[(indices[2 + (indices[0])])];
    fptype ret = 0;
    fptype integret = 0;
//    fptype bernkn = 0;
//    fptype integbernkn = 0;
    int ipar=2;
//    fptype ifactni=1;
//    fptype ifactik=1;
    fptype xmin=p[(indices[numParams-1])];
    fptype xmax=p[(indices[numParams])];
    x=(x-xmin)/(xmax-xmin);
//     printf("BernsteinTestPdf => limit xmin= %f xmax= %f\n",xmin,xmax);
//     return 0;


//     for(int i = 2; i < numParams; ++i) {
//         ret += (p[(indices[i])]) * pow(x, lowestDegree + i - 2);
//     }
    
      float k;
//      float i;
      for(k = 0; k <=maxDegree; ++k) {
       if (ipar>numParams-1){
        printf("Error in BernsteinTestPdf => ipar=%d > numParams=%d\n",ipar,numParams);
        return 0;
       }
       ret      += (p[(indices[ipar])]) * device_BernsteinTestkn_func(x,maxDegree,k);
       integret += (p[(indices[ipar])]) * device_BernsteinTestkn_intg(x,maxDegree,k);
//       printf("BernsteinTestPdf => %f integral = %f k=%d numparam=%d par=%f\n",ret,integret,k,numParams,(p[(indices[ipar])]));
       ipar++;
      }
//       printf("BernsteinTestPdf => %f int = %f\n",ret,integret);
    if(ret<1.E-30)  return 1.E-30;
    return ret/integret/(xmax-xmin);
//    return 0.;
}

/* __device__ fptype device_OffsetBernsteinTest(fptype *evt, fptype *p, unsigned int *indices) {
    int numParams    = (indices[0]);
    int lowestDegree = (indices[1]);

    fptype x = evt[(indices[2 + numParams])];
    x -= (p[(indices[numParams])]);
    fptype ret = 0;

    for(int i = 2; i < numParams; ++i) {
        ret += (p[(indices[i])]) * pow(x, lowestDegree + i - 2);
    }

    return ret*ret;
}
*/
//=================================================================================================================================== 
//
//================ device_MultiBernsteinTest ======================================
// 
//=================================================================================================================================== 
__device__ fptype device_MultiBernsteinTest(fptype *evt, fptype *p, unsigned int *indices) {
//       if ((0 == THREADIDX) && (0 == BLOCKIDX)){
//        printf("==================================================\n");
//       }
// //     struct bernVal bernknvalx;
//     struct bernVal bernknvaly;
//     struct bernVal bernknvalz;
    int numObservables = (indices[(indices[0]) + 1]);
    int maxDegree1      = (indices[1]);
    int maxDegree2      = (indices[2]);
    int maxDegree3      = (indices[3]);
//      if ((0 == THREADIDX) && (0 == BLOCKIDX)){
//      printf("MultiBernsteinTest 0=%d 0+1=%d 0+2=%d 0+3=%d\n",indices[(indices[0])],indices[(indices[0])+1],indices[(indices[0])+2],indices[(indices[0])+3]);
//      printf("MultiBernsteinTest numObservables=%d maxDegree1=%d maxDegree2=%d maxDegree3=%d\n",numObservables,indices[1],indices[2],indices[3]);
//      }
    if (numObservables!=3) {
     printf("device_MultiBernsteinTest error: Max Number of Observables is = 3!!! numObservables = %d\n",numObservables);
     return -100;
    }
 
    fptype x    = (evt[(indices[2 + (indices[0]) ])]); // x, y, z...
    fptype y    = (evt[(indices[2 + (indices[0]) + 1])]); // x, y, z...
    fptype z    = (evt[(indices[2 + (indices[0]) + 2])]); // x, y, z...
//      if ((0 == THREADIDX) && (0 == BLOCKIDX)){
//      printf("MultiBernsteinTest x=%5.15f y=%5.15f z=%5.15f %d %d %d\n",x,y,z,numObservables,indices[1],indices[2],indices[3]);
//      }
    fptype xmin = (p[(indices[4 ])]);
    fptype xdif = (p[(indices[5 ])])-(p[(indices[4 ])]);
    x=(x-xmin)/xdif;
    fptype ymin = (p[(indices[6])]);
    fptype ydif = (p[(indices[7])])-(p[(indices[6])]);
    y=(y-ymin)/ydif;
    fptype zmin = (p[(indices[8])]);
    fptype zdif = (p[(indices[9])])-(p[(indices[8])]);
    z=(z-zmin)/zdif;
    
//        if ((0 == THREADIDX) && (0 == BLOCKIDX)){
// 	printf("MultiBernsteinTest xmin=%5.15f xmax = %5.15f\n",xmin,xdif);
// 	printf("MultiBernsteinTest ymin=%5.15f ymax = %5.15f\n",ymin,ydif); 
// 	printf("MultiBernsteinTest zmin=%5.15f zmax = %5.15f\n",zmin,zdif);
// 	printf("MultiBernsteinTest [0,1] x=%5.15f y=%5.15f z=%5.15f \n",x,y,z);
//        
//        }
       double sx[10]={0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
   
       double sy[10]={0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
   
       double sz[10]={0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
       
       int enne = max(maxDegree3,max(maxDegree1,maxDegree2));
      
       if ( enne>9) {
        printf("device_MultiBernsteinTest error: Max(Numbers of degree) > 9 = %d\n",enne);
        return 0.0;
       }
       
       
       sx[0]=1.0;
       sy[0]=1.0;
       sz[0]=1.0;
//       for( int i = 1; i <= 9 ; ++i){
       for( int i = 1; i <= enne ; ++i){
        sx[i]= sx[i-1]*(1.-x);
        sy[i]= sy[i-1]*(1.-y);
        sz[i]= sz[i-1]*(1.-z);
       }
       int ipar =4 + 2*numObservables;
       int ipa0 =ipar;
//       int kk = 0;
//       int ii = 0;
//       int jj = 0;
       fptype func =0.0;
       fptype bernknvalx = 0.0;
       fptype bernknvaly = 0.0;
       fptype bernknvalz = 0.0;
       fptype intg_1 =0.0;
       fptype tx = 1.;
       for(int i = 0; i <= maxDegree1 ; ++i) {
         bernknvalx =  device_coeffbinomial_ber(maxDegree1,i)*tx*sx[maxDegree1-i];
//       jj = 0;
         fptype ty = 1.;
         for(int j = 0; j <= maxDegree2 ; ++j) {
	  bernknvaly =  device_coeffbinomial_ber(maxDegree2,j)*ty*sy[maxDegree2-j];
//	  std::cout<<"func = par["<<ipar<<"]*x^"<<kk<<"*y^"<<jj<<std::endl;
//          ii = 0;
          fptype tz = 1.;
          for(int k = 0; k <= maxDegree3 ; ++k) {
// 	   fptype bernknvalx =  device_coeffbinomial_ber(maxDegree1,i)*tx*pow(1.0-x,maxDegree1-i);
// 	   fptype bernknvaly =  device_coeffbinomial_ber(maxDegree2,j)*ty*pow(1.0-y,maxDegree2-j);
//  	   fptype bernknvalz =  device_coeffbinomial_ber(maxDegree3,k)*tz*pow(1.0-z,maxDegree3-k);
	   if (k==0) ipa0=ipar;
	   bernknvalz =  device_coeffbinomial_ber(maxDegree3,k)*tz*sz[maxDegree3-k];
//	   std::cout<<"func = par["<<ipar<<"]*x^"<<ii<"*y^"<<jj<<"*z^"<<kk<<std::endl;
//        	   fptype bernknvalx =  device_BernsteinTestkn_func(x,maxDegree1,i);
//     	   fptype bernknvaly =  device_BernsteinTestkn_func(y,maxDegree2,j);
//     	   fptype bernknvalz =  device_BernsteinTestkn_func(z,maxDegree3,k);
// 	   fptype bernknintx =  device_BernsteinTestkn_intg(x,maxDegree1,i);
// 	   fptype bernkninty =  device_BernsteinTestkn_intg(y,maxDegree2,j);
// 	   fptype bernknintz =  device_BernsteinTestkn_intg(z,maxDegree3,k);
//            func +=(p[(indices[ipar])])*bernknvalx*bernknvaly*bernknvalz;
//            intg +=(p[(indices[ipar])])*bernknintx*bernkninty*bernknintz;
 	   if(k==maxDegree3){
            func   +=(p[(indices[ipa0])])*bernknvalx*bernknvaly*bernknvalz;
            intg_1 +=(p[(indices[ipa0])]);
	   }else{
            func   +=(p[(indices[ipar])])*bernknvalx*bernknvaly*bernknvalz;
            intg_1 +=(p[(indices[ipar])]);
	    ipar++;
	   } 
// 	    if ((0 == THREADIDX) && (0 == BLOCKIDX)){
//  	     printf("MultiBernsteinTest  par = %f       \n",(p[(indices[ipar])]));
// 	     printf("MultiBernsteinTest  par = %f       B_(%d,%d,%d) = %f intg=%f\n",(p[(indices[ipar])]),ii,jj,kk,bernknvalx,bernknintx);
// 	    } 

//        if ((0 == THREADIDX) && (0 == BLOCKIDX)){
// 	printf("MultiBernsteinTest MaxDegree=%d coefficient = %f   number = %d\n",maxDegree,(p[(indices[ipar])]),ipar-2-2*numObservables);
//        } 
	   
//           ii = (jj+kk+ii<maxDegree?++ii:0);
	   tz*=z;
	  }
//          jj = (jj+kk+ii<maxDegree?++jj:0);
	  
	   ty*=y;
	 
         }
//         kk= (jj+kk+ii<maxDegree?++kk:0);
	   tx*=x;
       }
//       return  func*func;
//       return  func/(intg);
//      return  func/(intg)/xdif/ydif/zdif;
//      return  func/(intg)/xdif/ydif/zdif;
      
// okkio se commenti qui sotto, cosi' non e' normalizzato!!!!!!!!!!!!!! 
      intg_1 = (1.0+maxDegree1)*(1.0+maxDegree2)*(1.0+maxDegree3)/intg_1;     
      intg_1 = intg_1/xdif*ydif*zdif;
      func=func*intg_1;
      if(func<1.E-30)  return 1.E-30;
      return  func;
 }
//
//=================================================================================================================================== 
//
//================ device_MultiBinBernsteinTest ======================================
// 
//=================================================================================================================================== 
__device__ fptype device_MultiBinBernsteinTest(fptype *evt, fptype *p, unsigned int *indices) {
//        if ((0 == THREADIDX) && (0 == BLOCKIDX)){
//         printf("==================================================\n");
//        }
// //     struct bernVal bernknvalx;
//     struct bernVal bernknvaly;
//     struct bernVal bernknvalz;
    int numObservables  = (indices[(indices[0]) + 1]);
    int maxDegree1      = (indices[1]);
    int maxDegree2      = (indices[2]);
    int maxDegree3      = (indices[3]);
//      if ((0 == THREADIDX) && (0 == BLOCKIDX)){
//      printf("MultiEffiBernsteinTest 0=%d 0+1=%d 0+2=%d 0+3=%d\n",indices[(indices[0])],indices[(indices[0])+1],indices[(indices[0])+2],indices[(indices[0])+3]);
//      printf("MultiEffiBernsteinTest numObservables=%d maxDegree1=%d maxDegree2=%d maxDegree3=%d\n",numObservables,indices[1],indices[2],indices[3]);
//      }
    if ( (numObservables)!=3) {
     printf("device_MultiBinBernsteinTest error: Max Number of Observables is = 3!!! numObservables = %d\n",numObservables);
     return -100;
    }
 
    fptype x    = (evt[(indices[2 + (indices[0]) ])]); // x, y, z...
    fptype y    = (evt[(indices[2 + (indices[0]) + 1])]); // x, y, z...
    fptype z    = (evt[(indices[2 + (indices[0]) + 2])]); // x, y, z...

    fptype xBinw =(p[(indices[10])]);
    fptype yBinw =(p[(indices[11])]);
    fptype zBinw =(p[(indices[12])]);
//    fptype zBinw =atan2(0.0,-1.0)/5.;
//    fptype zBinw =0.02;

    fptype xmin = (p[(indices[4 ])]);
    fptype xdif = (p[(indices[5 ])])-(p[(indices[4 ])]);

    fptype xLeft  = ((x-xBinw/2.)-xmin)/xdif;
    fptype xRight = ((x+xBinw/2.)-xmin)/xdif;
//    x=(x-xmin)/xdif;
    fptype ymin = (p[(indices[6])]);
    fptype ydif = (p[(indices[7])])-(p[(indices[6])]);
    fptype yLeft  = ((y-yBinw/2.)-ymin)/ydif;
    fptype yRight = ((y+yBinw/2.)-ymin)/ydif;
//    y=(y-ymin)/ydif;
    fptype zmin = (p[(indices[8])]);
    fptype zdif = (p[(indices[9])])-(p[(indices[8])]);
    fptype zLeft  = ((z-zBinw/2.)-zmin)/zdif;
    fptype zRight = ((z+zBinw/2.)-zmin)/zdif;
//    z=(z-zmin)/zdif;
    
       int ipar =4 + 3*(numObservables);
       int ipa0 =ipar;
//       int kk = 0;
//       int ii = 0;
//       int jj = 0;
       fptype ret   =0;
//       fptype intg =0;
       for(int i = 0; i <= maxDegree1 ; ++i) {
//       jj = 0;
         for(int j = 0; j <= maxDegree2 ; ++j) {
//	  std::cout<<"func = par["<<ipar<<"]*x^"<<kk<<"*y^"<<jj<<std::endl;
//          ii = 0;
          for(int k = 0; k <= maxDegree3 ; ++k) {

	   if (k==0) ipa0=ipar;
           fptype bernknintgbinx = device_BernsteinTestkn_intgBin(xLeft,xRight,maxDegree1,i);
           fptype bernknintgbiny = device_BernsteinTestkn_intgBin(yLeft,yRight,maxDegree2,j);
           fptype bernknintgbinz = device_BernsteinTestkn_intgBin(zLeft,zRight,maxDegree3,k);
	   if(k==maxDegree3){
            ret   +=(p[(indices[ipa0])])*bernknintgbinx*bernknintgbiny*bernknintgbinz;
	   }else{
            ret   +=(p[(indices[ipar])])*bernknintgbinx*bernknintgbiny*bernknintgbinz;
	    ipar++;
	   } 
//	   
//           ii = (jj+kk+ii<maxDegree?++ii:0);
	  }
//          jj = (jj+kk+ii<maxDegree?++jj:0);
	  
	 
         }
//         kk= (jj+kk+ii<maxDegree?++kk:0);
       }
    ret=ret/(xBinw*yBinw*zBinw);
    if(ret<1.E-30) ret = 1.E-30;

   return ret;

 }
//=================================================================================================================================== 
//
//================ device_MultiBinBernsteinWrongTest ======================================
// 
//=================================================================================================================================== 
__device__ fptype device_MultiBinBernsteinWrongTest(fptype *evt, fptype *p, unsigned int *indices) {
//        if ((0 == THREADIDX) && (0 == BLOCKIDX)){
//         printf("==================================================\n");
//        }
// //     struct bernVal bernknvalx;
//     struct bernVal bernknvaly;
//     struct bernVal bernknvalz;
    int numObservables  = (indices[(indices[0]) + 1]);
    int maxDegree1      = (indices[1]);
    int maxDegree2      = (indices[2]);
    int maxDegree3      = (indices[3]);
//      if ((0 == THREADIDX) && (0 == BLOCKIDX)){
//      printf("MultiEffiBernsteinTest 0=%d 0+1=%d 0+2=%d 0+3=%d\n",indices[(indices[0])],indices[(indices[0])+1],indices[(indices[0])+2],indices[(indices[0])+3]);
//      printf("MultiEffiBernsteinTest numObservables=%d maxDegree1=%d maxDegree2=%d maxDegree3=%d\n",numObservables,indices[1],indices[2],indices[3]);
//      }
    if ( (numObservables)!=3) {
     printf("device_MultiBinBernsteinWrongTest error: Max Number of Observables is = 3!!! numObservables = %d\n",numObservables);
     return -100;
    }
 
    fptype x    = (evt[(indices[2 + (indices[0]) ])]); // x, y, z...
    fptype y    = (evt[(indices[2 + (indices[0]) + 1])]); // x, y, z...
    fptype z    = fabs(evt[(indices[2 + (indices[0]) + 2])]); // x, y, z...

    fptype xBinw =(p[(indices[10])]);
    fptype yBinw =(p[(indices[11])]);
    fptype zBinw =(p[(indices[12])]);
//    fptype zBinw =atan2(0.0,-1.0)/5.;
//    fptype zBinw =0.02;

    fptype xmin = (p[(indices[4 ])]);
    fptype xdif = (p[(indices[5 ])])-(p[(indices[4 ])]);

    fptype xLeft  = ((x-xBinw/2.)-xmin)/xdif;
    fptype xRight = ((x+xBinw/2.)-xmin)/xdif;
//    x=(x-xmin)/xdif;
    fptype ymin = (p[(indices[6])]);
    fptype ydif = (p[(indices[7])])-(p[(indices[6])]);
    fptype yLeft  = ((y-yBinw/2.)-ymin)/ydif;
    fptype yRight = ((y+yBinw/2.)-ymin)/ydif;
//    y=(y-ymin)/ydif;
    fptype zmin = (p[(indices[8])]);
    fptype zdif = (p[(indices[9])])-(p[(indices[8])]);
    fptype zLeft  = ((z-zBinw/2.)-zmin)/zdif;
    fptype zRight = ((z+zBinw/2.)-zmin)/zdif;
//    z=(z-zmin)/zdif;
    
       int ipar =4 + 3*(numObservables);
       int ipa0 = ipar;
//       int kk = 0;
//       int ii = 0;
//       int jj = 0;
       fptype ret   =0;
//       fptype intg =0;
       for(int i = 0; i <= maxDegree1 ; ++i) {
//       jj = 0;
         for(int j = 0; j <= maxDegree2 ; ++j) {
//	  std::cout<<"func = par["<<ipar<<"]*x^"<<kk<<"*y^"<<jj<<std::endl;
//          ii = 0;
          for(int k = 0; k <= maxDegree3 ; ++k) {

	   if (k==0) ipa0=ipar;
           fptype bernknintgbinx = device_BernsteinTestkn_intgBin(xLeft,xRight,maxDegree1,i);
           fptype bernknintgbiny = device_BernsteinTestkn_intgBin(yLeft,yRight,maxDegree2,j);
           fptype bernknintgbinz = device_BernsteinTestkn_intgBin(zLeft,zRight,maxDegree3,k);
	   if(k==maxDegree3){
            ret   +=(p[(indices[ipa0])])*bernknintgbinx*bernknintgbiny*bernknintgbinz;
	   }else{
            ret   +=(p[(indices[ipar])])*bernknintgbinx*bernknintgbiny*bernknintgbinz;
	    ipar++;
	   } 
//           ii = (jj+kk+ii<maxDegree?++ii:0);
	  }
//          jj = (jj+kk+ii<maxDegree?++jj:0);
	  
	 
         }
//         kk= (jj+kk+ii<maxDegree?++kk:0);
       }
    if(ret<1.E-30) ret = 1.E-30;
 //   ret=ret/(xBinw*yBinw*zBinw);

   return ret;

 }
// device_MultiBinBernsteinWrongTest
//=================================================================================================================================== 
//
//================ device_MultiAdaptBernsteinTest ======================================
// 
//=================================================================================================================================== 
__device__ fptype device_MultiAdaptBernsteinTest(fptype *evt, fptype *p, unsigned int *indices) {
//         if ((0 == THREADIDX) && (0 == BLOCKIDX)){
// 	         printf("==================================================\n");
// 		 
//         }
    int numObservables  = (indices[(indices[0]) + 1]);
    int maxDegree1      = (indices[1]);
    int maxDegree2      = (indices[2]);
    int maxDegree3      = (indices[3]);
    if ( (numObservables-3)!=3) {
     printf("device_MultiAdaptBernsteinTest error: Max Number of Observables is = 3!!! numObservables = %d\n",numObservables-3);
     return 0.0;
    }
 
    fptype x     = (evt[(indices[2 + (indices[0])    ])]); // x, y, z...
    fptype y     = (evt[(indices[2 + (indices[0]) + 1])]); // x, y, z...
    fptype z     = (evt[(indices[2 + (indices[0]) + 2])]); // x, y, z...
//     fptype reco  = (evt[(indices[2 + (indices[0]) + 3])]); // x, y, z...
//     fptype gene  = (evt[(indices[2 + (indices[0]) + 4])]); // x, y, z...
     fptype xBinw = (evt[(indices[2 + (indices[0]) + 3])]); // x, y, z...
     fptype yBinw = (evt[(indices[2 + (indices[0]) + 4])]); // x, y, z...
     fptype zBinw = (evt[(indices[2 + (indices[0]) + 5])]); // x, y, z...
//    fptype xBinw = 2./25.; // x, y, z...
//    fptype yBinw = 2./25.; // x, y, z...
//    fptype zBinw = 2*3.1415926535/25.; // x, y, z...
//      if ((0 == THREADIDX) && (0 == BLOCKIDX)){
//      if ((THREADIDX==0) && (BLOCKIDX==0)){
//        printf("MultiEffiBernsteinTest x=%f y=%f z=%f xBinw=%f yBinw=%f zBinw=%f\n",x,y,z,xBinw,yBinw,zBinw);
//        printf("MultiEffiBernsteinTest numObservables=%d maxDegree1=%d maxDegree2=%d maxDegree3=%d\n",numObservables,indices[1],indices[2],indices[3]);
//        }
//     if (gene < reco || gene<0.0) {
//         printf("device_MultiAdaptBernsteinTest error: gene=%f < reco=%f THREADIDX==%d BLOCKIDX==%d\n",gene,reco,THREADIDX,BLOCKIDX);
//         return 0;
//     }
//    fptype nmax = 0;
    
// non se po' fa'?    if (reco==0.0) return exp( -1.0E30);
//    if (gene> nmax) nmax = gene;
//      if ((0 == THREADIDX) && (0 == BLOCKIDX)){
//      printf("MultiEffiBernsteinTest x=%5.15f y=%5.15f z=%5.15f %d %d %d\n",x,y,z,numObservables,indices[1],indices[2],indices[3]);
//      }    

//     fptype xBinw =(p[(indices[10])]);
//     fptype yBinw =(p[(indices[11])]);
//     fptype zBinw =(p[(indices[12])]);
//    fptype zBinw =atan2(0.0,-1.0)/5.;
//    fptype zBinw =0.02;

    fptype xmin = (p[(indices[4 ])]);
    fptype xdif = (p[(indices[5 ])])-(p[(indices[4 ])]);

    fptype xLeft  = ((x-xBinw/2.)-xmin)/xdif;
    fptype xRight = ((x+xBinw/2.)-xmin)/xdif;
//    x=(x-xmin)/xdif;
    fptype ymin = (p[(indices[6])]);
    fptype ydif = (p[(indices[7])])-(p[(indices[6])]);
    fptype yLeft  = ((y-yBinw/2.)-ymin)/ydif;
    fptype yRight = ((y+yBinw/2.)-ymin)/ydif;
//    y=(y-ymin)/ydif;
    fptype zmin = (p[(indices[8])]);
    fptype zdif = (p[(indices[9])])-(p[(indices[8])]);
    fptype zLeft  = ((z-zBinw/2.)-zmin)/zdif;
    fptype zRight = ((z+zBinw/2.)-zmin)/zdif;
//    z=(z-zmin)/zdif;
    
//  	if ( (47 == THREADIDX) && (0 == BLOCKIDX)){
//         printf("==================================================\n");
// //   printf("EffiBernsteinTestPdf THREADIDX==%d BLOCKIDX==%d\n",THREADIDX,BLOCKIDX);
//   	 printf("MultiEffiBernsteinTest x=%5.15f y=%5.15f z=%5.15f\n",x,y,z);
// //  	 printf("MultiEffiBernsteinTest xmin=%5.15f xdif = %5.15f\n",xmin,xdif);
// //  	 printf("MultiEffiBernsteinTest ymin=%5.15f ydif = %5.15f\n",ymin,ydif);
//   	 printf("MultiEffiBernsteinTest zmin=%5.15f zdif = %5.15f\n",zmin,zdif);
// //  	 printf("MultiEffiBernsteinTest xLeft=%5.15f xRight = %5.15f\n",xLeft,xRight);
// //  	 printf("MultiEffiBernsteinTest yLeft=%5.15f yRight = %5.15f\n",yLeft,yRight);
//   	 printf("MultiEffiBernsteinTest zLeft=%5.15f zRight = %5.15f\n",zLeft,zRight);
//  	}
       int ipar = 4 + numObservables*2; 
       int ipa0 = ipar;
       //conteggio indici: gli interi [numero Obs + 3 gradi del poly (maxDeg1,2,3) = 4] + num obs(vars+bins) + num limiti (6)
//       int kk = 0;
//       int ii = 0;
//       int jj = 0;
       fptype ret  =0.0;
       fptype intg_1 =0.;
//       fptype mu   =0.0;
//       fptype intg_1 =0;
       for(int i = 0; i <= maxDegree1 ; ++i) {
//       jj = 0;
         for(int j = 0; j <= maxDegree2 ; ++j) {
//	  std::cout<<"func = par["<<ipar<<"]*x^"<<kk<<"*y^"<<jj<<std::endl;
//          ii = 0;
          for(int k = 0; k <= maxDegree3 ; ++k) {
//	   std::cout<<"func = par["<<ipar<<"]*x^"<<ii<<"*y^"<<jj<<"*z^"<<kk<<std::endl;
//  	   fptype bernknvalx =  device_coeffbinomial_ber(maxDegree,ii)*pow(x,ii)*pow(1.0-x,maxDegree-ii);
//  	   fptype bernknvaly =  device_coeffbinomial_ber(maxDegree,jj)*pow(x,jj)*pow(1.0-x,maxDegree-jj);
//  	   fptype bernknvalz =  device_coeffbinomial_ber(maxDegree,kk)*pow(x,kk)*pow(1.0-x,maxDegree-kk);
//         fptype bernknvalx =  device_EffiBernsteinTestkn_func(x,maxDegree1,i);
//     	   fptype bernknvaly =  device_EffiBernsteinTestkn_func(y,maxDegree2,j);
//     	   fptype bernknvalz =  device_EffiBernsteinTestkn_func(z,maxDegree3,k);
// 	   fptype bernknintx =  device_EffiBernsteinTestkn_intg(maxDegree1,i);
// 	   fptype bernkninty =  device_EffiBernsteinTestkn_intg(maxDegree2,j);
// 	   fptype bernknintz =  device_EffiBernsteinTestkn_intg(maxDegree3,k);
//            func +=(p[(indices[ipar])])*bernknvalx*bernknvaly*bernknvalz;
//            intg +=(p[(indices[ipar])])*bernknintx*bernkninty*bernknintz;

	   if (k==0) ipa0=ipar;
           fptype bernknintgbinx = device_BernsteinTestkn_intgBin(xLeft,xRight,maxDegree1,i);
           fptype bernknintgbiny = device_BernsteinTestkn_intgBin(yLeft,yRight,maxDegree2,j);
           fptype bernknintgbinz = device_BernsteinTestkn_intgBin(zLeft,zRight,maxDegree3,k);
//           mu   +=(p[(indices[ipar])])*bernknintgbinx/(xBinw);
//           mu   +=(p[(indices[ipar])])*bernknintgbiny/(yBinw);
//           mu   +=(p[(indices[ipar])])*bernknintgbinz/(zBinw);
//           mu   +=(p[(indices[ipar])])*bernknintgbinx*bernknintgbiny*bernknintgbinz/(xBinw*yBinw*zBinw);
 	   if(k==maxDegree3){
            ret    +=(p[(indices[ipa0])])*bernknintgbinx*bernknintgbiny*bernknintgbinz;
            intg_1 +=(p[(indices[ipa0])]);
 	   }else{
            ret    +=(p[(indices[ipar])])*bernknintgbinx*bernknintgbiny*bernknintgbinz;
            intg_1 +=(p[(indices[ipar])]);
	    ipar++;
	   } 
//           intg_1 +=(p[(indices[ipar])]);
//  	if ( (47 == THREADIDX) && (0 == BLOCKIDX)){
// //  	 printf("MultiEffiBernsteinTest bernknintgbinx=%5.15f\n",bernknintgbinx);
// //  	 printf("MultiEffiBernsteinTest bernknintgbiny=%5.15f\n",bernknintgbiny);
//   	 printf("MultiEffiBernsteinTest bernknintgbinz=%5.15f\n",bernknintgbinz);
// 	} 
//           intg +=(p[(indices[ipar])]);
//           intg +=(p[(indices[ipar])])*bernknintx*bernkninty*bernknintz;
//   	    if ((0 == THREADIDX) && (0 == BLOCKIDX)){
//    	     printf("MultiEffiBernsteinTest  par = %f   ipar=%d    \n",(p[(indices[ipar])]),ipar);
// // // 	     printf("MultiEffiBernsteinTest  par = %f       B_(%d,%d,%d) = %f intg=%f\n",(p[(indices[ipar])]),ii,jj,kk,bernknvalx,bernknintx);
//   	    } 

//        if ((0 == THREADIDX) && (0 == BLOCKIDX)){
// 	printf("MultiEffiBernsteinTest MaxDegree=%d coefficient = %f   number = %d\n",maxDegree,(p[(indices[ipar])]),ipar-2-2*numObservables);
//        } 
	   
//           ii = (jj+kk+ii<maxDegree?++ii:0);
	  }
//          jj = (jj+kk+ii<maxDegree?++jj:0);
	  
	 
         }
//         kk= (jj+kk+ii<maxDegree?++kk:0);
       }
      if(ret<1.E-30) ret = 1.E-30;
//       intg_1 = (1.0+maxDegree1)*(1.0+maxDegree2)*(1.0+maxDegree3)/intg_1;
//       intg_1 = intg_1/(xdif*ydif*zdif);
//       ret=ret*intg_1/(xBinw*yBinw*zBinw);
//      ret=ret/(xBinw*yBinw*zBinw);// se commenti questo non hai piu' la densita' di probabilita' sul bin, ma (direttamente) la probabilita'
      intg_1 = (1.0+maxDegree1)*(1.0+maxDegree2)*(1.0+maxDegree3)/intg_1;
      ret = ret*intg_1;
      return ret;
 }

__device__ device_function_ptr ptr_to_BernsteinTest		    = device_BernsteinTest;
__device__ device_function_ptr ptr_to_MultiBernsteinTest	    = device_MultiBernsteinTest;
__device__ device_function_ptr ptr_to_MultiBinBernsteinTest	    = device_MultiBinBernsteinTest;
__device__ device_function_ptr ptr_to_MultiBinBernsteinWrongTest    = device_MultiBinBernsteinWrongTest;
__device__ device_function_ptr ptr_to_MultiAdaptBernsteinTest       = device_MultiAdaptBernsteinTest;

// Constructor for single-variate BernsteinTest, with optional zero point.
// __host__ BernsteinTestPdf::BernsteinTestPdf(std::string n, Observable _x, std::vector<Variable> weights, unsigned int lowestDegree)
//     : GooPdf(n, _x) {
//     std::vector<unsigned int> pindices;
//     pindices.push_back(lowestDegree);
// 
//     for(auto &weight : weights) {
//         pindices.push_back(registerParameter(weight));
//     }
// 
//     GET_FUNCTION_ADDR(ptr_to_BernsteinTest);
// 
//     initialize(pindices);
// }

//=========================================================== 
//Constructor for single-variate BernsteinTest.
//=========================================================== 
__host__ BernsteinTestPdf::BernsteinTestPdf(std::string n, Observable _x, std::vector<Variable> weights,std::vector<Variable> limits, unsigned int maxDegree)
    : GooPdf(n, _x) {
    std::vector<unsigned int> pindices;
    pindices.push_back(maxDegree);

    for(auto &weight : weights) {
        pindices.push_back(registerParameter(weight));
    }
    for(auto &limit : limits) {
        pindices.push_back(registerParameter(limit));
    }

     GET_FUNCTION_ADDR(ptr_to_BernsteinTest);
//    GET_FUNCTION_ADDR(ptr_to_OffsetBernsteinTest);

    initialize(pindices);
}
//=========================================================== 
// Constructor for multivariate BernsteinTest.
//=========================================================== 
 __host__ BernsteinTestPdf::BernsteinTestPdf(std::string n,
				       std::vector<Observable> obses,
				       std::vector<Variable> coeffs,
				       std::vector<Variable> limits,
				       unsigned int maxDegree1,
				       unsigned int maxDegree2,
				       unsigned int maxDegree3 )
        : GooPdf(n) {
     unsigned int numParameters = 1;
 
      size_t limit = 0;
      cudaDeviceGetLimit(&limit, cudaLimitStackSize);
      printf("cudaLimitStackSize: %u\n", (unsigned)limit);
      cudaDeviceSetLimit(cudaLimitStackSize, 2*limit);
      cudaDeviceGetLimit(&limit, cudaLimitStackSize);
      printf("cudaLimitStackSize: %u\n", (unsigned)limit);
      printf("GooFit::Multivariate BernsteinTest, num obs:%d\n",obses.size());
     // For 1 observable, equal to n = maxDegree + 1.
     // For two, n*(n+1)/2, ie triangular number. This generalises:
     // 3: Pyramidal number n*(n+1)*(n+2)/(3*2)
     // 4: Hyperpyramidal number n*(n+1)*(n+2)*(n+3)/(4*3*2)
     // ...
     for(unsigned int i = 0; i < obses.size(); ++i) {
	 registerObservable(obses[i]);
//	 numParameters *= (maxDegree + 1 + i);
     }
//  
//      for(int i = observables.size(); i > 1; --i)
// 	 numParameters /= i;
 
//     int j=1;
//     numParameters = pow((maxDegree+1),coeffs.size());
     numParameters = (maxDegree1+1)*(maxDegree2+1)*(maxDegree3);
//     numParameters = (maxDegree1+1)*(maxDegree2+1)*(maxDegree3+1);
     while(numParameters > coeffs.size()) {
	 char varName[100];
	 sprintf(varName, "%s_extra_coeff_%i", getName().c_str(), static_cast<int>(coeffs.size()));
 
	 coeffs.emplace_back(varName, 10.,0.00001,0.,500.);
 
	 std::cout << "Warning: " << getName() << " created dummy variable " << varName
		   << "  to account for all terms.\n";
     }
 
     while(limits.size() < 2*obses.size()) {
	 char varName[100];
	 sprintf(varName, "%s_extra_limits_%i", getName().c_str(), static_cast<int>(limits.size()));
	 limits.emplace_back(varName, 0);
     }
 
     std::vector<unsigned int> pindices;
     pindices.push_back(maxDegree1);
     pindices.push_back(maxDegree2);
     pindices.push_back(maxDegree3);
 
     for(auto &limit : limits) {
	 pindices.push_back(registerParameter(limit));
     }
 
     for(auto &coeff : coeffs) {
	 pindices.push_back(registerParameter(coeff));
     }
 
     GET_FUNCTION_ADDR(ptr_to_MultiBernsteinTest);
     initialize(pindices);
 }
//=========================================================== 
// Constructor for multivariate BIN BernsteinTest.
//=========================================================== 
 __host__ BernsteinTestPdf::BernsteinTestPdf(std::string n,
				     std::vector<Observable> obses,
				     std::vector<Variable> coeffs,
				     std::vector<Variable> limits,
				     std::vector<Variable> binws,
				     unsigned int maxDegree1,
				     unsigned int maxDegree2,
				     unsigned int maxDegree3 )
        : GooPdf(n) {
     unsigned int numParameters = 1;
      size_t limit = 0;
      cudaDeviceGetLimit(&limit, cudaLimitStackSize);
      printf("cudaLimitStackSize: %u\n", (unsigned)limit);
      cudaDeviceSetLimit(cudaLimitStackSize, 2*limit);
      cudaDeviceGetLimit(&limit, cudaLimitStackSize);
      printf("cudaLimitStackSize: %u\n", (unsigned)limit);
      printf("GooFit::Multivariate Bin BernsteinTest, num obs:%d\n",obses.size());
 
     // For 1 observable, equal to n = maxDegree + 1.
     // For two, n*(n+1)/2, ie triangular number. This generalises:
     // 3: Pyramidal number n*(n+1)*(n+2)/(3*2)
     // 4: Hyperpyramidal number n*(n+1)*(n+2)*(n+3)/(4*3*2)
     // ...
     for(unsigned int i = 0; i < obses.size(); ++i) {
	 registerObservable(obses[i]);
//	 numParameters *= (maxDegree + 1 + i);
     }
//  
//      for(int i = observables.size(); i > 1; --i)
// 	 numParameters /= i;
 
//     int j=1;
//     numParameters = pow((maxDegree+1),coeffs.size());
     numParameters = (maxDegree1+1)*(maxDegree2+1)*(maxDegree3);
//     numParameters = (maxDegree1+1)*(maxDegree2+1)*(maxDegree3+1);
     while(numParameters > coeffs.size()) {
	 char varName[100];
	 sprintf(varName, "%s_extra_coeff_%i", getName().c_str(), static_cast<int>(coeffs.size()));
 
	 coeffs.emplace_back(varName, 10.,0.00001,0.,500.);
 
	 std::cout << "Warning: " << getName() << " created dummy variable " << varName
		   << "  to account for all terms.\n";
     }
 
     while(limits.size() < 2*(obses.size()-2)) {
	 char varName[100];
	 sprintf(varName, "%s_extra_limits_%i", getName().c_str(), static_cast<int>(limits.size()));
	 limits.emplace_back(varName, 0);
     }
 
     std::vector<unsigned int> pindices;
     pindices.push_back(maxDegree1);
     pindices.push_back(maxDegree2);
     pindices.push_back(maxDegree3);
 
     for(auto &limit : limits) {
	 pindices.push_back(registerParameter(limit));
     }
     for(auto &binw : binws) {
	 pindices.push_back(registerParameter(binw));
     }
 
     for(auto &coeff : coeffs) {
	 pindices.push_back(registerParameter(coeff));
     }
 
     GET_FUNCTION_ADDR(ptr_to_MultiBinBernsteinTest);
     initialize(pindices);
 }
 
//=========================================================== 
// Constructor for multivariate BIN BernsteinTest WRONG.
//=========================================================== 
 __host__ BernsteinTestPdf::BernsteinTestPdf(std::string n,
				     std::vector<Observable> obses,
				     std::vector<Variable>   coeffs,
				     std::vector<Variable>   limits,
				     std::vector<Variable>   binws,
				     unsigned int maxDegree1,
				     unsigned int maxDegree2,
				     unsigned int maxDegree3,
				     unsigned int dummy1,
				     unsigned int dummy2 )
        : GooPdf(n) {
     unsigned int numParameters = 1;
      size_t limit = 0;
      cudaDeviceGetLimit(&limit, cudaLimitStackSize);
      printf("cudaLimitStackSize: %u\n", (unsigned)limit);
      cudaDeviceSetLimit(cudaLimitStackSize, 2*limit);
      cudaDeviceGetLimit(&limit, cudaLimitStackSize);
      printf("cudaLimitStackSize: %u\n", (unsigned)limit);
      printf("GooFit::Multivariate Bin Wrong BernsteinTest, num obs:%d\n",obses.size());
 
     // For 1 observable, equal to n = maxDegree + 1.
     // For two, n*(n+1)/2, ie triangular number. This generalises:
     // 3: Pyramidal number n*(n+1)*(n+2)/(3*2)
     // 4: Hyperpyramidal number n*(n+1)*(n+2)*(n+3)/(4*3*2)
     // ...
     for(unsigned int i = 0; i < obses.size(); ++i) {
	 registerObservable(obses[i]);
//	 numParameters *= (maxDegree + 1 + i);
     }
//  
//      for(int i = observables.size(); i > 1; --i)
// 	 numParameters /= i;
 
//     int j=1;
//     numParameters = pow((maxDegree+1),coeffs.size());
     numParameters = (maxDegree1+1)*(maxDegree2+1)*(maxDegree3);
//     numParameters = (maxDegree1+1)*(maxDegree2+1)*(maxDegree3+1);
     while(numParameters > coeffs.size()) {
	 char varName[100];
	 sprintf(varName, "%s_extra_coeff_%i", getName().c_str(), static_cast<int>(coeffs.size()));
 
	 coeffs.emplace_back(varName, 10.,0.00001,0.,500.);
 
	 std::cout << "Warning: " << getName() << " created dummy variable " << varName
		   << "  to account for all terms.\n";
     }
 
     while(limits.size() < 2*(obses.size()-2)) {
	 char varName[100];
	 sprintf(varName, "%s_extra_limits_%i", getName().c_str(), static_cast<int>(limits.size()));
	 limits.emplace_back(varName, 0);
     }
 
     std::vector<unsigned int> pindices;
     pindices.push_back(maxDegree1);
     pindices.push_back(maxDegree2);
     pindices.push_back(maxDegree3);
 
     for(auto &limit : limits) {
	 pindices.push_back(registerParameter(limit));
     }
     for(auto &binw : binws) {
	 pindices.push_back(registerParameter(binw));
     }
 
     for(auto &coeff : coeffs) {
	 pindices.push_back(registerParameter(coeff));
     }
 
     GET_FUNCTION_ADDR(ptr_to_MultiBinBernsteinWrongTest);
     initialize(pindices);
 }
//=========================================================== 
// Constructor for multivariate Adaptive BernsteinTest.
//=========================================================== 
 __host__ BernsteinTestPdf::BernsteinTestPdf(std::string n,
				       std::vector<Observable> obses,
				       std::vector<Variable> coeffs,
				       std::vector<Variable> limits,
				       unsigned int maxDegree1,
				       unsigned int maxDegree2,
				       unsigned int maxDegree3,
				       unsigned int dummy )
        : GooPdf(n) {
     unsigned int numParameters = 1;
 //      size_t limit = 0;
//       cudaDeviceGetLimit(&limit, cudaLimitStackSize);
//       printf("cudaLimitStackSize: %u\n", (unsigned)limit);
//       cudaDeviceSetLimit(cudaLimitStackSize, 2*limit);
//       cudaDeviceGetLimit(&limit, cudaLimitStackSize);
//       printf("cudaLimitStackSize: %u\n", (unsigned)limit);
      printf("GooFit::Multivariate Adaptive BernsteinTest, num obs:%d\n",obses.size());
     // For 1 observable, equal to n = maxDegree + 1.
     // For two, n*(n+1)/2, ie triangular number. This generalises:
     // 3: Pyramidal number n*(n+1)*(n+2)/(3*2)
     // 4: Hyperpyramidal number n*(n+1)*(n+2)*(n+3)/(4*3*2)
     // ...
     for(unsigned int i = 0; i < obses.size(); ++i) {
	 registerObservable(obses[i]);
//	 numParameters *= (maxDegree + 1 + i);
     }
//  
//      for(int i = observables.size(); i > 1; --i)
// 	 numParameters /= i;
 
//     int j=1;
//     numParameters = pow((maxDegree+1),coeffs.size());
     numParameters = (maxDegree1+1)*(maxDegree2+1)*(maxDegree3);
//     numParameters = (maxDegree1+1)*(maxDegree2+1)*(maxDegree3+1);
     while(numParameters > coeffs.size()) {
	 char varName[100];
	 sprintf(varName, "%s_extra_coeff_%i", getName().c_str(), static_cast<int>(coeffs.size()));
 
	 coeffs.emplace_back(varName, 10.,0.00001,0.,500.);
 
	 std::cout << "Warning: " << getName() << " created dummy variable " << varName
		   << "  to account for all terms.\n";
     }
 
     while(limits.size() < 2*obses.size()) {
	 char varName[100];
	 sprintf(varName, "%s_extra_limits_%i", getName().c_str(), static_cast<int>(limits.size()));
	 limits.emplace_back(varName, 0);
     }
 
     std::vector<unsigned int> pindices;
     pindices.push_back(maxDegree1);
     pindices.push_back(maxDegree2);
     pindices.push_back(maxDegree3);
 
     for(auto &limit : limits) {
	 pindices.push_back(registerParameter(limit));
     }
 
     for(auto &coeff : coeffs) {
	 pindices.push_back(registerParameter(coeff));
     }
 
     GET_FUNCTION_ADDR(ptr_to_MultiAdaptBernsteinTest);
     initialize(pindices);
 }
 
//========================================================================
//======================================================================== 
 __host__ fptype BernsteinTestPdf::integrate(fptype lo, fptype hi) const {
       return 1.0;
 }

} // namespace GooFit
