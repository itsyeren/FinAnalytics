# USA 514 Stocks Prices NASDAQ NYSE

The USA 514 Stocks Prices Dataset provides historical Open, High, Low, Close, and Volume (OHLCV) prices of 514 stocks traded in the United States financial markets NASDAQ NYSE. You can use price movements and trading volumes for stock price predictions.

### ~11 Gb of market data for you and your analysis with NN or other methods 

Here is the link to kaggle dataset [USA 514 Stocks Prices NASDAQ NYSE](https://www.kaggle.com/datasets/olegshpagin/usa-stocks-prices-ohlcv) (weekly updates)

#### 4626 CSV files for MN1, W1, D1, H4, H1, M30, M15, M10 and M5 timeframes

```

Tesla D1 Daily timeframe
     datetime  open  high   low  close    volume
0  2010-06-29  1.23  1.66  1.16   1.59  48396981
1  2010-06-30  1.72  2.02  1.55   1.57  50220666
2  2010-07-01  1.66  1.72  1.34   1.46  24136347
3  2010-07-02  1.53  1.54  1.24   1.28  14933295
4  2010-07-06  1.33  1.33  1.05   1.06  19921698

... ... ... ... ... ... ... ...

        datetime    open    high     low   close    volume
3493  2024-03-08  178.73  182.73  174.70  175.41  67134329
3494  2024-03-09  175.41  175.41  175.32  175.39     33414
3495  2024-03-11  175.45  182.87  174.80  177.81  66873611
3496  2024-03-12  177.77  179.43  172.41  177.47  67692765
3497  2024-03-13  173.05  176.05  169.15  169.49  77267229



Tesla H1 Hourly timeframe
              datetime  open  high   low  close    volume
0  2010-06-29 18:00:00  1.23  1.26  1.16   1.20  17642793
1  2010-06-29 19:00:00  1.20  1.23  1.20   1.22   5992710
2  2010-06-29 20:00:00  1.22  1.27  1.22   1.27   5145171
3  2010-06-29 21:00:00  1.27  1.46  1.27   1.46   9751716
4  2010-06-29 22:00:00  1.46  1.66  1.40   1.59   9864591

... ... ... ... ... ... ... ...

                  datetime    open    high     low   close    volume
24791  2024-03-13 19:00:00  172.16  172.44  170.85  170.95   6582167
24792  2024-03-13 20:00:00  170.95  172.05  170.70  170.88   7240207
24793  2024-03-13 21:00:00  170.88  171.23  170.06  170.25   8665781
24794  2024-03-13 22:00:00  170.25  171.04  169.15  169.43  12279126
24795  2024-03-13 23:00:00  169.43  169.49  169.40  169.49     27612

```

If you want to download actual data - on today for example, then you can use python code from my github.

**Feel free to write in comments which stocks prices you are interested most, may be it is possible to download their prices too.** 

tickers = ['CE.US', 'WELL.US', 'GRMN.US', 'IEX.US', 'CAG.US', 'BEN.US', 'ATO.US', 'WY.US', 'TSCO.US', 'COR.US', 'MOS.US', 'SWKS.US', 'ORCL.US', 'URI.US', 'INCY.US', 'MPC.US', 'HD.US', 'PPG.US', 'NUE.US', 'DDOG.US', 'HSIC.US', 'CAT.US', 'HSY.US', 'MKTX.US', 'CCEP.US', 'GWW.US', 'LEN.US', 'IFF.US', 'GL.US', 'MDB.US', 'SNPS.US', 'KR.US', 'DVN.US', 'SYY.US', 'USB.US', 'DRI.US', 'PARA.US', 'FMC.US', 'UBER.US', 'WRK.US', 'DLR.US', 'SO.US', 'AMGN.US', 'MA.US', 'STT.US', 'BWA.US', 'KVUE.US', 'GFS.US', 'BBY.US', 'BK.US', 'MRVL.US', 'VFC.US', 'EIX.US', 'ADSK.US', 'ZBH.US', 'MU.US', 'HUBB.US', 'PEAK.US', 'CVX.US', 'CPB.US', 'GILD.US', 'BXP.US', 'DD.US', 'MCD.US', 'KDP.US', 'GE.US', 'PKG.US', 'HST.US', 'WTW.US', 'XOM.US', 'ED.US', 'SPG.US', 'PFG.US', 'LVS.US', 'FAST.US', 'ROST.US', 'TTD.US', 'CNC.US', 'PGR.US', 'CMI.US', 'TEAM.US', 'MELI.US', 'BKR.US', 'EBAY.US', 'CPRT.US', 'MSFT.US', 'HOLX.US', 'ABBV.US', 'AMZN.US', 'FE.US', 'WYNN.US', 'KMI.US', 'APA.US', 'CRWD.US', 'DPZ.US', 'EQT.US', 'NOC.US', 'TAP.US', 'ETR.US', 'T.US', 'OMC.US', 'MTCH.US', 'TRMB.US', 'EXPE.US', 'DTE.US', 'PNR.US', 'LH.US', 'ALL.US', 'CTRA.US', 'VMC.US', 'XRAY.US', 'NWS.US', 'GOOGL.US', 'WEC.US', 'BIIB.US', 'LLY.US', 'BMY.US', 'STE.US', 'NI.US', 'MKC.US', 'AMT.US', 'CFG.US', 'LW.US', 'HIG.US', 'ETSY.US', 'AON.US', 'ULTA.US', 'DVA.US', 'LKQ.US', 'MPWR.US', 'TEL.US', 'FICO.US', 'CVS.US', 'CMA.US', 'NVDA.US', 'TDG.US', 'AWK.US', 'PSA.US', 'FOXA.US', 'ON.US', 'ODFL.US', 'NVR.US', 'ROP.US', 'TFX.US', 'HLT.US', 'EXPD.US', 'FOX.US', 'D.US', 'AMAT.US', 'AZO.US', 'DLTR.US', 'TT.US', 'SBUX.US', 'JNJ.US', 'HAS.US', 'DASH.US', 'NRG.US', 'JNPR.US', 'BIO.US', 'AMD.US', 'NFLX.US', 'VLTO.US', 'BRO.US', 'REGN.US', 'WRB.US', 'LRCX.US', 'SYK.US', 'MCO.US', 'CSGP.US', 'TROW.US', 'ETN.US', 'RTX.US', 'CRM.US', 'SIRI.US', 'UPS.US', 'HES.US', 'RSG.US', 'PEP.US', 'MET.US', 'HON.US', 'IQV.US', 'JPM.US', 'DG.US', 'CBRE.US', 'NDSN.US', 'DOW.US', 'SBAC.US', 'TSN.US', 'IT.US', 'WM.US', 'TPR.US', 'IBM.US', 'CHTR.US', 'HAL.US', 'ROL.US', 'FDS.US', 'SHW.US', 'EW.US', 'RJF.US', 'APH.US', 'AIZ.US', 'ZBRA.US', 'SRE.US', 'CTAS.US', 'PXD.US', 'MTD.US', 'NOW.US', 'MAS.US', 'FFIV.US', 'ELV.US', 'SYF.US', 'CSCO.US', 'APTV.US', 'FI.US', 'LHX.US', 'DAL.US', 'DFS.US', 'COST.US', 'LOW.US', 'SJM.US', 'ABNB.US', 'KHC.US', 'CCL.US', 'PM.US', 'WBD.US', 'PANW.US', 'CCI.US', 'ANET.US', 'ASML.US', 'CMS.US', 'PDD.US', 'MRNA.US', 'ACGL.US', 'LIN.US', 'CMG.US', 'PRU.US', 'ISRG.US', 'DGX.US', 'TSLA.US', 'AAPL.US', 'FDX.US', 'RVTY.US', 'BKNG.US', 'MLM.US', 'UNP.US', 'HBAN.US', 'CB.US', 'NTRS.US', 'FLT.US', 'TJX.US', 'AEE.US', 'HII.US', 'ES.US', 'BAC.US', 'CDNS.US', 'ABT.US', 'QRVO.US', 'AXON.US', 'ANSS.US', 'TFC.US', 'BA.US', 'JKHY.US', 'STZ.US', 'CTSH.US', 'TDY.US', 'CZR.US', 'MDLZ.US', 'A.US', 'CDW.US', 'ARE.US', 'XEL.US', 'STX.US', 'DIS.US', 'VTRS.US', 'IPG.US', 'WAT.US', 'AVB.US', 'FITB.US', 'BSX.US', 'OKE.US', 'BLDR.US', 'TRGP.US', 'OTIS.US', 'AME.US', 'CINF.US', 'HCA.US', 'PWR.US', 'ALB.US', 'DHR.US', 'VLO.US', 'FSLR.US', 'CDAY.US', 'CHRW.US', 'AVGO.US', 'NWSA.US', 'MS.US', 'AJG.US', 'HPQ.US', 'WBA.US', 'PPL.US', 'J.US', 'KEY.US', 'TECH.US', 'VICI.US', 'AMCR.US', 'ILMN.US', 'EA.US', 'FCX.US', 'MO.US', 'RL.US', 'TYL.US', 'ENPH.US', 'ADBE.US', 'BDX.US', 'EPAM.US', 'BF.B.US', 'AVY.US', 'CTLT.US', 'TER.US', 'APD.US', 'GS.US', 'ZS.US', 'KMX.US', 'INTC.US', 'MNST.US', 'PHM.US', 'PH.US', 'PFE.US', 'DXCM.US', 'MGM.US', 'EMN.US', 'STLD.US', 'BR.US', 'RF.US', 'TXT.US', 'WDAY.US', 'WST.US', 'CMCSA.US', 'GNRC.US', 'MCHP.US', 'EQR.US', 'ROK.US', 'MCK.US', 'MDT.US', 'PSX.US', 'BAX.US', 'FTNT.US', 'PNC.US', 'FIS.US', 'HRL.US', 'ICE.US', 'EL.US', 'KEYS.US', 'FTV.US', 'ADI.US', 'NCLH.US', 'IR.US', 'META.US', 'CME.US', 'DHI.US', 'V.US', 'MMM.US', 'MSI.US', 'DUK.US', 'MRO.US', 'TMO.US', 'JBL.US', 'CSX.US', 'AES.US', 'UHS.US', 'GEHC.US', 'UDR.US', 'CAH.US', 'LUV.US', 'DOV.US', 'PAYC.US', 'EVRG.US', 'CL.US', 'IP.US', 'F.US', 'HWM.US', 'K.US', 'C.US', 'REG.US', 'AIG.US', 'CTVA.US', 'ITW.US', 'AMP.US', 'TTWO.US', 'PG.US', 'KLAC.US', 'SCHW.US', 'AAL.US', 'PNW.US', 'WAB.US', 'MSCI.US', 'O.US', 'CLX.US', 'LDOS.US', 'YUM.US', 'L.US', 'ZION.US', 'AXP.US', 'COO.US', 'ALGN.US', 'ALLE.US', 'RCL.US', 'LYV.US', 'MMC.US', 'COF.US', 'CHD.US', 'PODD.US', 'NKE.US', 'SWK.US', 'GD.US', 'GPC.US', 'ADP.US', 'IVZ.US', 'BG.US', 'GEN.US', 'ESS.US', 'AFL.US', 'PCAR.US', 'CEG.US', 'NXPI.US', 'CRL.US', 'LNT.US', 'EOG.US', 'AKAM.US', 'JBHT.US', 'EXR.US', 'MOH.US', 'XYL.US', 'MTB.US', 'TGT.US', 'NTAP.US', 'ACN.US', 'CPT.US', 'FANG.US', 'ADM.US', 'PLD.US', 'CF.US', 'WMT.US', 'GM.US', 'POOL.US', 'TXN.US', 'CARR.US', 'EXC.US', 'FRT.US', 'INVH.US', 'HUM.US', 'NEE.US', 'UNH.US', 'LYB.US', 'EMR.US', 'AZN.US', 'NEM.US', 'WFC.US', 'MAR.US', 'PTC.US', 'CNP.US', 'WHR.US', 'ECL.US', 'KMB.US', 'EG.US', 'ZTS.US', 'BLK.US', 'PEG.US', 'TRV.US', 'IDXX.US', 'GIS.US', 'CI.US', 'IRM.US', 'BBWI.US', 'BALL.US', 'PAYX.US', 'QCOM.US', 'GLW.US', 'CBOE.US', 'VRSK.US', 'RMD.US', 'SPGI.US', 'KO.US', 'MRK.US', 'DE.US', 'WMB.US', 'MAA.US', 'BRK.B.US', 'EQIX.US', 'EFX.US', 'SPLK.US', 'TMUS.US', 'LMT.US', 'VZ.US', 'PYPL.US', 'HPE.US', 'MHK.US', 'NDAQ.US', 'KIM.US', 'AEP.US', 'JCI.US', 'OXY.US', 'WDC.US', 'VTR.US', 'GPN.US', 'BX.US', 'UAL.US', 'AOS.US', 'GOOG.US', 'VRSN.US', 'INTU.US', 'ORLY.US', 'NSC.US', 'VRTX.US', 'LULU.US', 'SNA.US', 'RHI.US', 'COP.US', 'PCG.US', 'SLB.US']
