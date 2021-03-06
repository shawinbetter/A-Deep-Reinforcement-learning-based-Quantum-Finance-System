//****************************************************************************************************************************
// Date        : 2021/05/14                                                                                               
// Created by  : Charm Zhang (4 May 2021 )                                                                              
// Name        : QPL Cacluation Program + Project Input Data        
//               VERSION NO: 1.0            
// Objective   : This program calculate the N(QPR) of 5 Financial Products
//               For Each Financial Product:
//               0) Cacluate All K values [K0 .. K20] using the following formula:
//                  K[eL] = pow((1.1924 + 33.2383*eL + 56.2169*eL*eL)/(1 + 43.6106 *eL),p3);
//               1) Read the Daily Time Series and extract (Date, O, H, L, C, V) m
//               2) Calculate Dally Price Return r(t)
//               3) Calculate quantum price return wavefunction Q(r)(size 100)
//               4) Evalutate lambda (L) value for the wavefunction Q(r)using F.D.M. at ground state
//                  L = abs((r0^2*Q0 - r1^2*Q1)/(r0^4*Q0 - r1^4*Q1))
//               5) Evaluate other related parameters:  
//                  - sigma  (std dev of Q)
//                  - maxQPR (max Quantum Price Return - for normalization)       
//               6) Once L is found, using Quartic Schrodinger Equation of Quantum Finance to find 
//                  all the 21 Quantum Price Energies (QFEL0 .. QFEL20). 
//                  Given by:
//                  (E(n)/(2n+1))^3 - (E(n)/(2n+1)) - K(n)^3 * L = 0
//                  where 
//                   K(n) = ((1.1924+33.2383n+56.2169n^2)/(1+43.6106n))^(1/3)
//               7) Solve the 21 Cubic Eqts in (6) and extract the +ve real roots as QFEL0 .. QFEL20.
//               8) Cacluate QPR(n)  = QFEL(n)/QFEL(0) n = [1 .. 20]
//               9) Cacluate NQPR(n) = 1 + 0.21*sigma*QPR(n);
//               10)Save TWO Level of datafiles
//                  1) For each financial product, save the QPL Table contains
//                     QFEL, QPR, NQPR for the first 21 energy levels
//                  2) For all financial product, create a QPL Summary table contains NQPR for all FP                                                      
//                     
//                                                                                                                           
//****************************************************************************************************************************

#property copyright "Copyright © 2019, DR. RAYMOND LEE"
#property link      "http://QFFC.ORG"
#property version   "1.10"
#property strict


// DEFINE DIRECTORIES
string      result_Directory = "input_data";
int         result_FileHandle;
string      result_FileName = "";

// DEFINE GLOBAL VARIABLES
int         maxELevel  = 2;              // Max Energy Level      
int         maxTP      = 5;              // Max no of Financial Product
int         maxTS      = 1024;            // Max no of Time Series Record
int         nTP=0;
double      p3=1.0/3.0;                   // Set 1/3 for MathPow

//DEFINE TIMING VARIABLES
uint        stime=0;
uint        etime=0;
uint        Gstime=0;
uint        Getime=0;
uint        tlapse=0;
uint        Gtlapse=0;

// DEFINE FINANCIAL PRODUCT RELATED VARIABLES
string      TPSymbol   = "";              // Current Trading duct Symbol

/*
string      TP_Code[10]={"XAUUSD","AUDUSD","EURCHF","EURUSD","GBPUSD",
                         "USDCHF","USDJPY","NZDUSD","EURGBP","XAGUSD"};
                         
int         TP_No[10]={1,2,3,4,5,6,7,8,9,10};

int         TP_nD[10]={2,5,5,5,5,5,3,5,5,3};
*/

//string      TP_Code[5]={"XAGUSD",  "Uber",   "USDJPY", "AUDUSD", "EURCHF"};

string      TP_Code[5]={"USDJPY",  "AUDUSD", "#GOOG", "US100_M1", "XRPUSD"};
                         
int         TP_No[5]={1,2,3,4,5};

// int         TP_nD[5]={3,5,3,5,5};


//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
//======================================================================================================================
// ****************************************
// LOOP OVER ALL TP

   stime       = GetTickCount();          // Get timer
   // Set Global Start Time
   Gstime       = GetTickCount();
   for(nTP=0; nTP<maxTP; nTP++){
         //*******************************************************************
         //
         // 0. Cacluate All K values [K0 .. K20] using the following formula:
         //
         //  K[eL] = pow((1.1924 + 33.2383*eL + 56.2169*eL*eL)/(1 + 43.6106 *eL),p3);
         //
         
         // Printout K List Header
         Print("Printout ALL K values K0 .. K20 for first 20 Energy Levels");
         
         int eL;
         double     K[2];              // K values in QP Schrodinger Eqt
         for(eL=0;eL<2;eL++){
            K[eL] = pow((1.1924 + 33.2383*eL + 56.2169*eL*eL)/(1 + 43.6106 *eL),p3);
            Print("Energy Level ",eL," K",eL," = ",K[eL]);
         }

         TPSymbol    = TP_Code[nTP];            // Get TP Symbol
         result_FileName      = TP_No[nTP] + "_"+TPSymbol+"Data.csv";
         FileDelete(result_Directory+"//"+result_FileName,FILE_COMMON);
         ResetLastError();
         result_FileHandle    = FileOpen(result_Directory+"//"+result_FileName,FILE_COMMON|FILE_READ|FILE_WRITE|FILE_CSV,',');
         // Write Header Line for result DataFile
         FileWrite(result_FileHandle,"OPEN", "CLOSE", "HIGH", "LOW", "VOLUME", "QPL", "QPL+", "QPL-", "MA5", "MA21", "MACD", "RSI", "BBup",
         "BBlow", "CCI", "Stoch", "ADX", "ATR");
         
         for(int day=0; day<1024; day++){
               int      d, TSsize, nQ, maxQno, nR, maxRno, tQno;
               double   auxR, maxQ, r0, r1, rn1;
               double   dr, Lup, Ldw, L, mu, sigma; 
               bool     bFound;
               
               // Variables used in Cardano's Method
               double   p, q, u, v;
               
               // Declare Time Series Array
               double     DT_CL[1024];
               double     DT_RT[1024];
               
               // Declare Array for Quantum Price Wavefunction 
               double     Q[100];            // Quantum Price Wavefunction
               double     NQ[100];           // Normalized Q[]
               double     r[100];            // r no
               
               // Declare ARRAY for QPL related arrays
               double     ALL_QFEL[5][2];  // Array contains QFEL  for all FPs
               double     ALL_QPR[5][2];   // Array contains QPR  for all FPs
               double     ALL_NQPR[5][2];  // Array contains NQPR for all FPs
               double     QFEL[2];           // QFEL for each FP
               double     QPR[2];            // QPR  for each FP
               double     NQPR[2];           // NQPR for each FP
               double     QPL_neg[2];
               double     QPL_pos[2];
               
               // Other indicators
               double     MA5;
               double     MA21;
               double     MACD;
               double     RSI;
               double     BBup;
               double     BBlow;
               double     CCI;
               double     Stoch;
               double     ADX;
               double     ATR;
               
               //======================================================================================================================
               // FIXME: PART 1

               
               //********************************************************************************************************
               //
               // 1. READ ALL Daily Time Series 
               //
               //********************************************************************************************************
               
               // Since iBars/Bars doesn't work, manually check TSsize
               TSsize      = 0;
               while(iTime(TPSymbol,PERIOD_D1,TSsize)>0 && (TSsize<maxTS)){
                  TSsize++;
               }
               Print(TPSymbol," SIZE: ",TSsize);
               
               // Using For LOOP to get all the time series data
               for(d=1;d<TSsize;d++){
                  DT_CL[d-1] = iClose(TPSymbol,PERIOD_D1,d+1);
                  DT_RT[d-1] = 1; //        
               }
               MA5 = iMA(TPSymbol,PERIOD_D1,5,8,MODE_SMA,PRICE_CLOSE,day+1);
               MA21 = iMA(TPSymbol,PERIOD_D1,21,8,MODE_SMA,PRICE_CLOSE,day+1);
               MACD = iMACD(TPSymbol,PERIOD_D1,12,26,9,PRICE_CLOSE,MODE_MAIN,day+1);
               RSI = iRSI(TPSymbol,PERIOD_D1,14,PRICE_CLOSE,day+1);
               BBup = iBands(TPSymbol,PERIOD_D1,14,2,0,PRICE_CLOSE,MODE_UPPER,day+1);
               BBlow = iBands(TPSymbol,PERIOD_D1,14,2,0,PRICE_CLOSE,MODE_LOWER,day+1);
               CCI = iCCI(TPSymbol,PERIOD_D1,12,PRICE_CLOSE,day+1);
               Stoch = iStochastic(TPSymbol,PERIOD_D1,5,3,3,MODE_SMA,0,MODE_MAIN,day+1);
               ADX = iADX(TPSymbol,PERIOD_D1,14,PRICE_CLOSE,MODE_MAIN,day+1);
               ATR = iATR(TPSymbol,PERIOD_D1,12,day+1);
               
               // Cacluate DT_RT[d], 
               //======================================================================================================================
               // FIXME: PART 2
               for(d=0;d<(TSsize-2);d++){
                   DT_RT[d] = (DT_CL[d+1] > 0) ? DT_CL[d]/DT_CL[d+1] : 1;
               }
               
               //======================================================================================================================
               
               // Close QP Detail Data File
               // FileClose(QPD_FileHandle);
            
               //******************************************************************
               //
               // 2. Calculate Mean (mu) and Standard Deviation (sigma) of return array
               //
               //*******************************************************************
               maxRno = TSsize - 2;
               
               // Calculate mean mu first
               mu = 0;
               //======================================================================================================================
               // FIXME: PART 3
               for (d=0;d<maxRno;d++)
               {
                 mu += DT_RT[d];
               }
               mu /= maxRno;
               
               //======================================================================================================================
               
               // Calculate STDEV sigma
               //======================================================================================================================
               // FIXME: PART 4
               sigma = 0;
               for(d=0;d<maxRno;d++){
                  sigma += pow(DT_RT[d] - mu, 2);
               }
               sigma = sqrt((sigma / maxRno));
               
               // Calculate dr where dr = 3*sigma/50
               dr = 3 * sigma / 50;
               
               Print("TP",nTP+1," ",TP_Code[nTP]," No of r = ",maxRno," mu = ",mu," sigma = ",sigma," dr = ",dr);
               
               //======================================================================================================================
               
               //******************************************************************
               //
               // 3. Generate the QP Wavefunction distribution 
               //
               //*******************************************************************
               
               auxR = 0;
               
               // Loop over all r form r - 50*dr to r + 50*dr and get the distribution function
               // Reset all the Q[] first
               for(nQ=0;nQ<100;nQ++){
                  Q[nQ] = 0;
               }
               
               // Loop over the maxRno to get the distribution
               tQno = 0;
               for(nR=0;nR<maxRno;nR++){
                  bFound = False;
                  nQ = 0;
                  auxR = 1 - (dr*50);
                  while(!bFound && (nQ < 100)){
                     if((DT_RT[nR] > auxR && (DT_RT[nR] <= (auxR + dr)))){
                        Q[nQ]++;
                        tQno++;
                        bFound = True;
                     }else{
                        nQ++;
                        auxR = auxR + dr;
                     }
                  }
               }
               
               // Write out the Qfile for Record
               auxR = 1 - (dr*50);
               for(nQ=0;nQ<100;nQ++){
                  r[nQ] = auxR;
                  NQ[nQ] = Q[nQ]/tQno;
                  auxR = auxR + dr;
               }
               
               
               // Find maxQ and maxQno
               //======================================================================================================================
               // FIXME: PART 5
               maxQ     = 0;
               maxQno   = 0;
               maxQno = ArrayMaximum(NQ,WHOLE_ARRAY,0);
               maxQ = NQ[maxQno];
               //======================================================================================================================
               
               
               // Printout the maxQ and maxQno
               Print("TP",nTP+1," ",TP_Code[nTP]," MaxQ= ",maxQ," maxQno=",maxQno," Total Qno=",tQno);
               
               
               //******************************************************************
               //
               // 4. Evaluate Lambda L for the QP Wavefuntion
               //
               //*******************************************************************
               //     
               // Given maxQno - i.e. ground state Q[0], r[0] = r[maxQno-dr]
               // We have Q[+1] = NQ[maxQno+1], r[+1] = r[maxQno]+(dr/2)
               //         Q[-1] = NQ[maxQno-1], r[-1] = r[maxQno]-(dr*1.5)
               // Apply F.D.M. into QP Sch Eqtuation
               // L = abs((r[-1]^2*Q[-1]-(r[+1]^2*Q[+1]))/(r[-1]^4*Q[-1]-(r[+1]^4*Q[+1])))
               
               r0    = r[maxQno] - (dr/2); //-
               r1    = r0 + dr;
               rn1   = r0 - dr;
               Lup   = (pow(rn1,2)*NQ[maxQno-1])-(pow(r1,2)*NQ[maxQno+1]);
               Ldw   = (pow(rn1,4)*NQ[maxQno-1])-(pow(r1,4)*NQ[maxQno+1]);
               L = MathAbs(Lup/Ldw);
               
               // Printout r0,Q0, r1, Q1, r-1 Q-1
               Print("TP",nTP+1," ",TP_Code[nTP]," r0 = ",r0," r1 = ",r1," r-1 = ",rn1," Q0 = ",NQ[maxQno]," Q1 = ",NQ[maxQno+1]," Q-1 = ",NQ[maxQno-1]," L = Lup/Ldw = ",Lup,"/",Ldw," = ",L);
               Print(TPSymbol,L);
               
               //******************************************************************
               //
               // 5. Using QP Schrodinger Eqt to FIND first 21 Energy Levels
               //
               //    By solving the Quartic Anharmonic Oscillator as cubic polynomial eqt
               //    of the form
               //
               //        a*x^3 + b*x^2 + c*x + d = 0
               //
               //    Using (Dasqupta et. al. 2007) QAHO solving equation:
               // 
               //    (E(n)/(2n+1))^3 - (E(n)/(2n+1)) - K(n)^3*L = 0
               //
               //    Solving the above Depressed Cubic Eqt using Cardano's Method
               //    
               //    Given    t^3 + p*t + q = 0
               //    Let      t = u + v
               //    Cardano's Method deduced that:
               //        u^3 = -q/2 + sqrt(q^2/4 + p^3/27)
               //        v^3 = -q/2 - sqrt(q^2/4 + p^3/27)
               //    The first cubic root (real root) will be:
               //
               //        t = u + v        
               //
               //    So, combining Cardano's Method into our QF Sch Eqt. 
               //    We have
               //    Substitue p = -(2n+1)^2;  q = -L(2n+1)^3*(K(n)^3) into the above equations to get the 
               //    real root
               //
               //*********************************************************************************************
               
               for(eL=0;eL<2;eL++){
                  p = -1 * pow((2*eL+1),2);
                  q = -1 * L * pow((2*eL+1),3)*pow(K[eL],3);
                  
                  // Apply Cardano's Method to find the real root of the depressed cubic equation
                  u = MathPow((-0.5*q + MathSqrt(((q*q/4.0) + (p*p*p/27.0)))),p3);
                  v = MathPow((-0.5*q - MathSqrt(((q*q/4.0) + (p*p*p/27.0)))),p3);
                  
                  // Store the QFEL
                  QFEL[eL] = u + v;
                  
                  // Printout the QF Energy Levels
                  Print("TP",nTP+1," ",TP_Code[nTP]," Energy Level",eL," QFEL = ",QFEL[eL]);
                  
               }
               
               // Evaluate ALL QPR values
               double CURR_OP = iOpen(TPSymbol,PERIOD_D1,day+1);
               double CURR_HI = iHigh(TPSymbol,PERIOD_D1,day+1);
               double CURR_LO = iLow(TPSymbol,PERIOD_D1,day+1);
               double CURR_CL = iClose(TPSymbol,PERIOD_D1,day+1);
               double CURR_VL = iVolume(TPSymbol,PERIOD_D1,day+1);
               
               for(eL=0;eL<2;eL++){
                  QPR[eL] = QFEL[eL]/QFEL[0];
                  NQPR[eL] = 1 + 0.21*sigma*QPR[eL];
                  QPL_neg[eL] = CURR_OP / NQPR[eL]; 
                  QPL_pos[eL] = CURR_OP * NQPR[eL]; 
                  
                  // Store into ALL QFEL, QPR, NQPR, into array
                  ALL_QFEL[nTP,eL] = QFEL[eL];
                  ALL_QPR[nTP,eL] = QPR[eL];
                  ALL_NQPR[nTP,eL] = NQPR[eL];
                  
               }  
               FileWrite(result_FileHandle,CURR_OP, CURR_CL, CURR_HI, CURR_LO, CURR_VL, QPL_pos[0], QPL_pos[1], QPL_neg[1], MA5, MA21, MACD,RSI,
               BBup, BBlow, CCI, Stoch, ADX, ATR);


         }
         FileClose(result_FileHandle);
   }
   // Check Global Time
   Getime   = GetTickCount();
   Gtlapse  = Getime - Gstime;
   
   // Output time taken
   Print("Total Time Taken : ",Gtlapse," msec");
   
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   
  }
//+------------------------------------------------------------------+
