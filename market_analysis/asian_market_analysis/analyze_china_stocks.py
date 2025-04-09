"""
Unified China US-Listed Stock Analysis and Grading
This script analyzes China stocks listed on US exchanges, breaking them into batches internally.
"""
import os
import time
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt # Removed as plotting is not used here
import yfinance as yf
from datetime import datetime
import re # Import regex for ticker parsing

# --- Adjust sys.path to allow direct script execution ---
import sys
# Get the directory containing the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the script's directory to the Python path
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
# --- End path adjustment ---

# Import China-specific functions
try:
    from asian_economic_quadrants import determine_chinese_economic_quadrant
    from db_utils_china import save_china_stock_analysis, get_china_db_stats
except ImportError as e:
    print(f"Error importing China-specific modules: {e}")
    # Provide dummy functions if imports fail
    def determine_chinese_economic_quadrant(*args, **kwargs):
        print("Warning: Using dummy determine_chinese_economic_quadrant.")
        return 'C', {}, {} # Return a default quadrant and empty dicts
    def save_china_stock_analysis(*args, **kwargs):
        print("Warning: Using dummy save_china_stock_analysis.")
        return True
    def get_china_db_stats(*args, **kwargs):
        print("Warning: Using dummy get_china_db_stats.")
        return {'china_stock_analysis_count': 0, 'china_economic_quadrant_count': 0, 'china_batches': [], 'china_grade_distribution': {}}


# Define batch size for processing
BATCH_SIZE = 25 # Reduced batch size due to potential API limits and data size

# Raw text data containing the list of Chinese stocks
CHINA_STOCK_LIST_TEXT = """
Skip to main content
Company or stock symbol...
/
Log In
Sign Up
Home
Stocks

Stock Screener
Stock Exchanges
Comparison Tool
Earnings Calendar
By Industry
Stock Lists
Top Analysts
Top Stocks
Corporate Actions
IPOs

ETFs

News
Trending
Articles
Market Movers

Market Newsletter
Stock Analysis Pro
Watchlist

Collapse
Home
»
Lists
Chinese Companies on The US Stock Market
A complete list of the Chinese companies that are listed on the US stock market. Many of them are traded as American Depositary Receipts (ADRs) and are also listed on a stock exchange in China.

Total Stocks
320
Total Market Cap
799.98B
Total Revenue
535.50B
320 Stocks
Filter results
Find...

Download

Indicators
Screener

1	BABA	Alibaba Group Holding Limited	129.33	-0.35%	292.21B	134.51B
2	NTES	NetEase, Inc.	104.94	-1.37%	68.24B	14.43B
3	JD	JD.com, Inc.	39.90	-2.90%	57.91B	158.76B
4	BIDU	Baidu, Inc.	89.80	-2.32%	30.87B	18.24B
5	PUK	Prudential plc	20.92	-4.21%	27.33B	12.26B
6	BEKE	KE Holdings Inc.	20.98	2.69%	25.33B	12.80B
7	LI	Li Auto Inc.	24.77	-2.94%	24.94B	19.79B
8	TME	Tencent Music Entertainment Group	14.32	0.21%	22.66B	3.89B
9	XPEV	XPeng Inc.	21.12	-	20.07B	5.60B
10	YUMC	Yum China Holdings, Inc.	52.63	-1.22%	19.71B	11.30B
11	ZTO	ZTO Express (Cayman) Inc.	19.11	-1.90%	15.25B	6.07B
12	FUTU	Futu Holdings Limited	97.49	-5.78%	13.45B	1.54B
13	YMM	Full Truck Alliance Co. Ltd.	12.53	-3.69%	13.10B	1.54B
14	HTHT	H World Group Limited	37.14	-0.40%	11.64B	3.27B
15	BZ	Kanzhun Limited	18.87	-0.68%	8.30B	1.01B
16	TAL	TAL Education Group	13.51	-1.46%	8.18B	2.07B
17	BILI	Bilibili Inc.	19.13	-1.03%	7.97B	3.68B
18	NIO	NIO Inc.	3.740	-1.58%	7.86B	9.01B
19	EDU	New Oriental Education & Technology Group Inc.	47.79	-1.65%	7.82B	4.82B
20	VIPS	Vipshop Holdings Limited	14.83	-3.32%	7.65B	14.85B
21	QFIN	Qifu Technology, Inc.	43.48	-5.21%	6.10B	2.35B
22	ZK	ZEEKR Intelligent Technology Holding Limited	23.70	0.55%	6.02B	10.40B
23	MNSO	MINISO Group Holding Limited	18.07	-6.23%	5.57B	2.33B
24	GDS	GDS Holdings Limited	24.41	-7.19%	4.75B	1.41B
25	ZLAB	Zai Lab Limited	36.69	0.71%	4.00B	398.99M
26	ATAT	Atour Lifestyle Holdings Limited	28.54	-2.96%	3.93B	992.99M
27	WRD	WeRide Inc.	12.88	-6.05%	3.66B	49.48M
28	KC	Kingsoft Cloud Holdings Limited	14.88	2.69%	3.43B	1.07B
29	ATHM	Autohome Inc.	27.32	-2.18%	3.31B	964.45M
30	HCM	HUTCHMED (China) Limited	15.25	-4.09%	2.65B	630.20M
31	FINV	FinVolution Group	9.73	-2.41%	2.53B	1.79B
32	LU	Lufax Holding Ltd	2.900	-3.97%	2.51B	4.67B
33	PONY	Pony AI Inc.	6.92	-8.47%	2.41B	75.03M
34	DGNX	Diginex Limited	102.55	7.96%	2.36B	1.18M
35	RLX	RLX Technology Inc.	1.840	-1.60%	2.34B	334.91M
36	MLCO	Melco Resorts & Entertainment Limited	5.32	-0.19%	2.23B	4.64B
37	WB	Weibo Corporation	9.17	-2.86%	2.23B	1.75B
38	HSAI	Hesai Group	16.83	-2.32%	2.15B	284.58M
39	IQ	iQIYI, Inc.	2.090	-4.57%	2.01B	4.00B
40	AAPG	Ascentage Pharma Group International	23.07	0.30%	2.00B	134.35M
41	VNET	VNET Group, Inc.	7.33	-9.73%	1.96B	1.13B
42	TUYA	Tuya Inc.	2.790	-9.12%	1.66B	298.62M
43	LX	LexinFintech Holdings Ltd.	9.87	-5.10%	1.62B	1.95B
44	SIMO	Silicon Motion Technology Corporation	44.09	-14.57%	1.49B	803.55M
45	TIGR	UP Fintech Holding Limited	8.30	-4.05%	1.46B	330.74M
46	LZMH	LZ Technology Holdings Limited	8.34	16.48%	1.27B	113.86M
47	EH	EHang Holdings Limited	18.95	-4.20%	1.20B	62.49M
48	DQ	Daqo New Energy Corp.	16.62	-8.43%	1.10B	1.03B
49	BGM	BGM Group Ltd.	10.87	-1.81%	1.06B	25.10M
50	MOMO	Hello Group Inc.	6.11	-2.55%	1.00B	1.45B
51	JKS	JinkoSolar Holding Co., Ltd.	17.53	-5.60%	931.80M	12.64B
52	DAO	Youdao, Inc.	7.75	-3.25%	926.27M	770.77M
53	LOT	Lotus Technology Inc.	1.310	-17.09%	888.50M	1.01B
54	UXIN	Uxin Limited	4.040	-6.05%	759.30M	232.10M
55	JFIN	Jiayin Group Inc.	14.27	-0.21%	756.77M	794.76M
56	GOTU	Gaotu Techedu Inc.	2.990	-1.97%	756.27M	623.85M
57	HUYA	HUYA Inc.	3.290	-0.60%	749.12M	832.86M
58	MSC	Studio City International Holdings Limited	3.875	-0.14%	746.28M	639.15M
59	XYF	X Financial	15.40	-4.76%	730.15M	804.45M
60	RERE	ATRenew Inc.	2.900	-3.65%	636.50M	2.24B
61	YRD	Yiren Digital Ltd.	7.01	-4.50%	609.41M	795.43M
62	NOAH	Noah Holdings Limited	9.18	-3.77%	606.68M	356.34M
63	ZKH	ZKH Group Limited	3.481	-1.39%	562.81M	1.20B
64	DDL	Dingdong (Cayman) Limited	2.570	-7.55%	558.51M	3.16B
65	ATGL	Alpha Technology Group Limited	33.34	9.39%	548.91M	1.59M
66	WDH	Waterdrop Inc.	1.400	-3.45%	516.52M	379.75M
67	DADA	Dada Nexus Limited	1.920	0.52%	498.83M	1.32B
68	QD	Qudian Inc.	2.620	-2.24%	496.24M	32.50M
69	LXEH	Lixiang Education Holding Co., Ltd.	22.54	-2.89%	432.02M	5.63M
70	CANG	Cango Inc.	4.080	-3.32%	423.46M	110.22M
71	SDA	SunCar Technology Group Inc.	3.830	-6.81%	416.00M	425.99M
72	ECX	ECARX Holdings Inc.	1.090	-3.11%	404.10M	761.91M
73	YSG	Yatsen Holding Limited	4.400	-7.56%	403.74M	464.91M
74	SOHU	Sohu.com Limited	12.63	-	379.73M	598.40M
75	DOGZ	Dogness (International) Corporation	28.83	-1.44%	367.02M	20.26M
76	AHG	Akso Health Group	1.010	-8.18%	363.94M	8.45M
77	SKBL	Skyline Builders Group Holding Limited	12.00	15.83%	362.70M	49.64M
78	ZH	Zhihu Inc.	4.140	-0.24%	353.30M	493.06M
79	PHH	Park Ha Biological Technology Co., Ltd.	12.78	-3.55%	337.06M	2.38M
80	RGC	Regencell Bioscience Holdings Limited	24.30	3.27%	316.21M	-
81	NIU	Niu Technologies	3.820	-7.95%	296.43M	450.51M
82	EM	Smart Share Global Limited	1.130	0.89%	291.75M	261.96M
83	LANV	Lanvin Group Holdings Limited	2.390	1.70%	280.39M	409.96M
84	WTF	Waton Financial Limited	5.86	-16.64%	278.83M	10.51M
85	QSG	QuantaSing Group Limited	5.21	-15.56%	266.30M	477.14M
86	ZJK	ZJK Industrial Co., Ltd.	4.260	-11.25%	261.48M	35.89M
87	XNET	Xunlei Limited	4.150	-10.17%	255.10M	323.14M
88	OCFT	OneConnect Financial Technology Co., Ltd.	6.66	-1.33%	252.73M	308.00M
89	GHG	GreenTree Hospitality Group Ltd.	2.390	-4.01%	242.69M	201.17M
90	HCAI	Huachen AI Parking Management Technology Holding Co., Ltd	7.40	-0.13%	234.77M	53.68M
91	ZYBT	Zhengye Biotechnology Holding Limited	4.790	-3.43%	227.00M	26.11M
92	EJH	E-Home Household Service Holdings Limited	1.230	13.89%	225.94M	50.69M
93	DOYU	DouYu International Holdings Limited	7.02	-3.17%	222.08M	585.12M
94	BDMD	Baird Medical Investment Holdings Limited	5.74	4.94%	205.37M	33.05M
95	BSII	Black Spade Acquisition II Co	10.13	0.30%	193.74M	-
96	MLGO	MicroAlgo Inc.	18.45	2.50%	184.00M	83.50M
97	NAMI	Jinxin Technology Holding Company	2.850	-3.39%	182.52M	56.13M
98	RITR	Reitar Logtech Holdings Limited	2.885	3.78%	180.15M	47.95M
99	FLX	BingEx Limited	2.500	9.17%	170.90M	612.15M
100	BZUN	Baozun Inc.	2.640	-4.00%	152.39M	1.29B
101	NCT	Intercont (Cayman) Limited	5.75	6.48%	152.38M	25.53M
102	ICG	Intchains Group Limited	2.190	-4.78%	131.36M	38.60M
103	HOLO	MicroCloud Hologram Inc.	0.706	-2.30%	125.03M	39.77M
104	CAAS	China Automotive Systems, Inc.	4.140	-1.19%	124.91M	650.94M
105	NTCL	NetClass Technology Inc	7.00	-4.11%	124.81M	10.10M
106	TOUR	Tuniu Corporation	1.050	-1.87%	123.60M	70.37M
107	CMCM	Cheetah Mobile Inc.	4.130	-2.13%	123.42M	110.54M
108	NCTY	The9 Limited	13.05	-3.26%	121.87M	24.29M
109	QH	Quhuo Limited	1.350	-4.26%	121.09M	493.47M
110	SRL	Scully Royalty Ltd.	8.12	-2.05%	120.36M	34.00M
111	EPWK	EPWK Holdings Ltd.	5.30	2.51%	118.30M	20.22M
112	KNDI	Kandi Technologies Group, Inc.	1.350	1.50%	116.20M	118.13M
113	CGTL	Creative Global Technology Holdings Limited	5.38	29.64%	115.33M	35.61M
114	NXTT	Next Technology Holding Inc.	0.264	-9.00%	115.13M	1.80M
115	IH	iHuman Inc.	2.160	-1.37%	112.36M	126.34M
116	STEC	Santech Holdings Limited	1.310	-	110.04M	260.47M
117	AZI	Autozi Internet Technology (Global) Ltd.	1.019	1.93%	108.02M	124.74M
118	ZJYL	Jin Medical International Ltd.	0.668	2.72%	104.57M	23.50M
119	VIOT	Viomi Technology Co., Ltd	1.500	-1.32%	102.31M	290.31M
120	GLAC	Global Lights Acquisition Corp	10.84	-	97.24M	-
121	MCTR	Ctrl Group Limited	6.33	-6.77%	96.85M	4.45M
122	MIMI	Mint Incorporation Limited	4.180	6.36%	96.19M	4.25M
123	PN	Skycorp Solar Group Ltd	3.430	-4.72%	92.61M	49.86M
124	HUHU	HUHUTECH International Group Inc.	4.300	-1.15%	91.05M	17.49M
125	HLP	Hongli Group Inc.	1.225	-4.30%	89.96M	14.05M
126	LUD	Luda Technology Group Limited	3.990	2.30%	89.77M	51.26M
127	LSE	Leishen Energy Holding Co., Ltd.	5.18	-4.43%	88.19M	69.07M
128	SY	So-Young International Inc.	0.861	-5.39%	86.95M	200.94M
129	WETO	Webus International Limited	3.920	-1.75%	86.24M	6.33M
130	THCH	TH International Limited	2.700	-11.48%	85.32M	207.03M
131	GMHS	Gamehaus Holdings Inc.	1.580	-7.60%	84.64M	145.24M
132	JVSA	JVSPAC Acquisition Corp.	10.72	-	82.40M	-
133	ASPC	A SPAC III Acquisition Corp.	10.09	-0.10%	81.27M	-
134	TROO	TROOPS, Inc.	0.708	-13.89%	79.63M	4.34M
135	SFWL	Shengfeng Development Limited	0.960	-7.69%	79.20M	504.16M
136	YHNA	YHN Acquisition I Limited	10.21	0.10%	79.13M	-
137	EURK	Eureka Acquisition Corp	10.33	0.10%	78.98M	-
138	NA	Nano Labs Ltd	4.695	-5.91%	77.46M	6.99M
139	ADAG	Adagene Inc.	1.640	1.23%	77.26M	103.20K
140	DSY	Big Tree Cloud Holdings Limited	1.350	6.30%	77.06M	7.32M
141	FVN	Future Vision II Acquisition Corp.	10.20	-	76.95M	-
142	RDAC	Rising Dragon Acquisition Corp.	10.20	-	76.49M	-
143	STG	Sunlands Technology Group	5.55	0.36%	75.03M	272.66M
144	XCH	XCHG Limited	1.260	2.44%	74.91M	39.07M
145	YAAS	Youxin Technology Ltd	2.100	-2.78%	70.46M	521.24K
146	EPSM	Epsium Enterprise Limited	5.27	5.82%	69.83M	17.91M
147	YI	111, Inc.	8.03	-5.19%	68.79M	1.97B
148	TOPW	Top Win International Limited	2.760	-27.94%	68.62M	15.23M
149	CCG	Cheche Group Inc.	0.880	-4.76%	68.51M	475.83M
150	CBAT	CBAK Energy Technology, Inc.	0.727	-4.37%	65.39M	176.61M
151	MATH	Metalpha Technology Holding Limited	1.700	-9.57%	64.54M	31.40M
152	YSXT	YSX Tech. Co., Ltd	2.630	2.33%	61.64M	66.14M
153	JG	Aurora Mobile Limited	10.06	-4.28%	60.22M	43.32M
154	YIBO	Planet Image International Limited	1.110	-5.93%	59.81M	149.83M
155	YXT	YXT.COM Group Holding Limited	0.980	3.16%	58.87M	45.37M
156	OCG	Oriental Culture Holding LTD	3.160	-7.06%	58.68M	1.23M
157	GMM	Global Mofy AI Limited	3.170	-7.85%	55.54M	41.36M
158	RCON	Recon Technology, Ltd.	1.888	6.09%	52.85M	9.00M
159	ELPW	Elong Power Holding Limited	0.794	-0.75%	52.57M	-
160	CNF	CNFinance Holdings Limited	0.732	0.29%	50.21M	109.79M
161	PRE	Prenetics Global Limited	4.000	-4.08%	48.87M	25.56M
162	PTHL	Pheton Holdings Ltd	3.400	-12.82%	48.45M	448.20K
163	MSW	Ming Shing Group Holdings Limited	3.700	-7.04%	48.01M	31.77M
164	BIYA	Baiya International Group Inc.	3.690	-28.90%	46.13M	13.67M
165	WIMI	WiMi Hologram Cloud Inc.	0.460	17.95%	45.18M	84.14M
166	TOP	TOP Financial Group Limited	1.210	-6.20%	44.81M	2.73M
167	UCL	uCloudlink Group Inc.	1.190	-4.80%	44.78M	91.64M
168	LSB	LakeShore Biopharma Co., Ltd	2.331	-0.39%	44.48M	79.42M
169	AIXI	Xiao-I Corporation	4.000	-5.44%	42.19M	65.64M
170	ZEPP	Zepp Health Corporation	2.800	-9.09%	41.99M	182.60M
171	JYD	Jayud Global Logistics Limited	0.429	24.25%	41.01M	82.43M
172	MASK	3 E Network Technology Group Limited	3.620	3.13%	40.73M	4.56M
173	BNR	Burning Rock Biotech Limited	3.950	-7.49%	40.52M	70.67M
174	CLIK	Click Holdings Limited	2.640	13.30%	39.34M	6.05M
175	LGCL	Lucas GC Limited	0.490	4.48%	38.20M	145.69M
176	SJ	Scienjoy Holding Corporation	0.880	-2.22%	36.82M	205.41M
177	UBXG	U-BX Technology Ltd.	3.730	0.54%	36.54M	51.60M
178	RAYA	Erayak Power Solution Group Inc.	1.300	8.33%	36.40M	22.82M
179	STAK	STAK Inc.	3.200	0.95%	36.00M	18.92M
180	ABLV	Able View Global Inc.	0.860	-4.44%	35.81M	130.09M
181	HKIT	Hitek Global Inc.	1.220	-4.69%	35.75M	3.45M
182	PLUT	Plutus Financial Group Limited	2.500	-1.96%	35.63M	952.97K
183	DIST	Distoken Acquisition Corporation	11.11	-4.63%	35.55M	-
184	ZBAO	Zhibao Technology Inc.	1.112	-4.99%	35.04M	25.27M
185	ZCMD	Zhongchao Inc.	1.380	8.66%	34.80M	17.41M
186	RAY	Raytech Holding Limited	1.970	4.79%	34.70M	9.94M
187	NCEW	New Century Logistics (BVI) Limited	1.550	-9.36%	33.33M	52.18M
188	CASI	CASI Pharmaceuticals, Inc.	2.125	-2.07%	32.92M	28.54M
189	GLE	Global Engine Group Holding Limited	1.750	1.74%	32.02M	6.33M
190	CLPS	CLPS Incorporation	1.136	-0.35%	31.79M	153.82M
191	JZXN	Jiuzi Holdings, Inc.	2.760	-3.50%	30.39M	1.40M
192	WAI	Top KingWin Ltd	0.165	-17.60%	30.15M	3.99M
193	NISN	Nisun International Enterprise Development Group Co., Ltd	6.42	-7.09%	29.20M	452.26M
194	ONEG	OneConstruction Group Limited	2.090	-4.13%	27.17M	60.90M
195	HPH	Highest Performances Holdings Inc.	0.109	-18.41%	26.97M	287.00M
196	AACG	ATA Creativity Global	0.866	-8.89%	26.80M	36.73M
197	SWIN	Solowin Holdings	1.655	-3.22%	26.77M	2.42M
198	EHGO	Eshallgo Inc.	1.050	5.00%	26.74M	15.39M
199	FENG	Phoenix New Media Limited	2.170	-3.36%	26.11M	96.41M
200	HUIZ	Huize Holding Limited	2.480	-6.06%	24.60M	171.11M
201	EPOW	Sunrise New Energy Co., Ltd.	0.900	-8.17%	24.29M	46.63M
202	ORIS	Oriental Rise Holdings Limited	1.080	-7.69%	23.77M	16.75M
203	LBGJ	Li Bang International Corporation Inc.	1.260	6.78%	23.62M	10.79M
204	MTEN	Mingteng International Corporation Inc.	3.760	-11.94%	23.34M	9.14M
205	INLF	INLIF Limited	1.600	-0.62%	23.20M	14.52M
206	UTSI	UTStarcom Holdings Corp.	2.450	5.60%	23.12M	10.88M
207	EBON	Ebang International Holdings Inc.	3.670	-10.27%	23.05M	3.59M
208	HUDI	Huadi International Group Co., Ltd.	1.570	-0.63%	22.42M	74.27M
209	CDTG	CDT Environmental Technology Investment Holdings Limited	2.000	7.53%	21.65M	31.43M
210	MTC	MMTec, Inc.	0.850	-10.11%	21.41M	-1.99M
211	DXST	Decent Holding Inc.	1.310	0.77%	21.29M	11.54M
212	IZM	ICZOOM Group Inc.	1.750	-1.13%	20.53M	177.93M
213	KUKE	Kuke Music Holding Limited	2.860	-2.72%	20.47M	14.78M
214	BTCT	BTC Digital Ltd.	3.870	-3.49%	20.33M	8.48M
215	CCM	Concord Medical Services Holdings Limited	4.500	8.96%	19.54M	64.91M
216	FEDU	Four Seasons Education (Cayman) Inc.	9.23	-7.98%	19.53M	27.97M
217	MOGU	MOGU Inc.	2.216	-2.38%	18.21M	19.80M
218	CCTG	CCSC Technology International Holdings Limited	1.530	1.32%	17.72M	16.46M
219	GDHG	Golden Heaven Group Holdings Ltd.	0.510	1.27%	17.61M	22.33M
220	WETH	Wetouch Technology Inc.	1.395	-2.45%	16.64M	41.02M
221	MGIH	Millennium Group International Holdings Limited	1.470	0.68%	16.54M	38.53M
222	CHR	Cheer Holding, Inc.	1.350	-2.17%	16.38M	147.20M
223	PT	Pintec Technology Holdings Limited	1.020	-1.92%	16.17M	4.48M
224	EDTK	Skillful Craftsman Education Technology Limited	1.010	0.75%	16.09M	624.37K
225	JFU	9F Inc.	1.350	-2.17%	15.89M	40.61M
226	ENFY	Enlightify Inc.	1.090	1.71%	15.75M	87.36M
227	DTSS	Datasea Inc.	2.070	-4.61%	15.50M	47.28M
228	QMMM	QMMM Holdings Limited	0.900	-3.25%	15.49M	2.70M
229	CREG	Smart Powerr Corp.	0.617	-1.74%	15.31M	-
230	XHG	XChange TEC.INC	0.740	-1.33%	15.19M	41.11M
231	SISI	Shineco, Inc.	0.875	-1.24%	15.03M	11.07M
232	JZ	Jianzhi Education Technology Group Company Limited	0.740	-10.79%	14.94M	50.81M
233	WXM	WF International Limited	2.080	-27.27%	14.35M	15.52M
234	XIN	Xinyuan Real Estate Co., Ltd.	2.470	-5.73%	14.05M	667.14M
235	SEED	Origin Agritech Limited	1.910	-2.55%	13.75M	16.16M
236	SFHG	Samfine Creation Holdings Group Limited	0.674	-1.14%	13.69M	19.01M
237	JDZG	JIADE Limited	0.554	0.51%	13.58M	2.33M
238	PWM	Prestige Wealth Inc.	0.425	-5.56%	13.45M	639.91K
239	YQ	17 Education & Technology Group Inc.	1.650	-1.20%	12.82M	25.92M
240	DUO	Fangdd Network Group Ltd.	0.297	6.25%	12.56M	37.35M
241	GLXG	Galaxy Payroll Group Limited	0.695	6.92%	12.52M	3.86M
242	TIRX	Tian Ruixiang Holdings Ltd	1.216	-2.72%	12.37M	3.22M
243	PGHL	Primega Group Holdings Limited	0.466	-7.20%	12.31M	17.17M
244	HKPD	Hong Kong Pharma Digital Technology Holdings Limited	1.110	-9.02%	12.21M	20.77M
245	FEBO	Fenbo Holdings Limited	1.080	-10.74%	11.95M	16.32M
246	STFS	Star Fashion Culture Holdings Limited	0.980	-2.00%	11.91M	14.97M
247	YYAI	Connexa Sports Technologies Inc.	0.810	-0.60%	11.80M	-
248	PETZ	TDH Holdings, Inc.	1.130	-4.24%	11.67M	3.28M
249	PSIG	PS International Group Ltd.	0.441	-8.13%	11.12M	112.32M
250	AIFU	AIX Inc.	0.194	-7.58%	11.00M	324.92M
251	JL	J-Long Group Limited	3.480	-5.69%	10.93M	32.83M
252	UCAR	U Power Limited	2.685	0.17%	10.81M	4.27M
253	SOS	SOS Limited	5.00	9.65%	10.50M	111.12M
254	MHUA	Meihua International Medical Technologies Co., Ltd.	0.316	-4.36%	10.07M	94.25M
255	ROMA	Roma Green Finance Limited	0.791	-2.28%	9.47M	1.03M
256	BHAT	Blue Hat Interactive Entertainment Technology	1.860	-13.49%	9.19M	40.46M
257	AIHS	Senmiao Technology Limited	0.870	-0.57%	9.15M	5.88M
258	NCI	Neo-Concept International Group Holdings Limited	0.448	-6.78%	9.10M	22.48M
259	WCT	Wellchange Holdings Company Limited	0.195	23.03%	9.02M	2.37M
260	CJET	Chijet Motor Company, Inc.	1.680	-1.18%	9.00M	10.28M
261	YJ	Yunji Inc.	1.775	0.28%	8.73M	66.96M
262	CLWT	Euro Tech Holdings Company Limited	1.125	-0.44%	8.68M	17.24M
263	CPOP	Pop Culture Group Co., Ltd	0.570	4.68%	8.52M	47.38M
264	MI	NFT Limited	2.060	-5.50%	8.41M	1.27M
265	SUGP	SU Group Holdings Limited	0.602	-7.14%	8.33M	23.45M
266	ATXG	Addentax Group Corp.	0.830	4.36%	8.13M	4.55M
267	INTJ	Intelligent Group Limited	0.616	-0.15%	8.09M	2.34M
268	KRKR	36Kr Holdings Inc.	4.020	0.25%	7.98M	31.66M
269	HIHO	Highway Holdings Limited	1.810	-1.09%	7.97M	7.35M
270	NAAS	NaaS Technology Inc.	0.599	-13.80%	7.92M	43.26M
271	WOK	WORK Medical Technology Group LTD	0.534	-7.13%	7.79M	11.51M
272	MFI	mF International Limited	0.581	-3.17%	7.70M	3.88M
273	GRFX	Graphex Group Limited	1.000	-9.09%	7.55M	24.18M
274	GURE	Gulf Resources, Inc.	0.685	3.04%	7.35M	12.80M
275	GSIW	Garden Stage Limited	0.470	-2.08%	7.34M	1.31M
276	LOBO	Lobo EV Technologies Ltd.	0.840	-15.99%	7.25M	19.47M
277	MEGL	Magic Empire Global Limited	1.430	-4.67%	7.24M	1.98M
278	RETO	ReTo Eco-Solutions, Inc.	3.700	2.92%	7.16M	3.84M
279	WAFU	Wah Fu Education Group Limited	1.590	-2.45%	7.01M	6.37M
280	ILAG	Intelligent Living Application Group Inc.	0.370	-0.72%	6.69M	8.98M
281	CLEU	China Liberal Education Holdings Limited	1.980	2.59%	6.67M	2.41M
282	RDGT	China Jo-Jo Drugstores, Inc.	1.070	-6.14%	6.61M	147.13M
283	PMAX	Powell Max Limited	0.435	-2.68%	6.36M	5.97M
284	ZKIN	ZK International Group Co., Ltd.	1.200	-4.38%	6.20M	108.20M
285	ANTE	AirNet Technology Inc.	0.415	-4.27%	5.94M	500.00K
286	CHSN	Chanson International Holding	0.211	-6.21%	5.77M	15.98M
287	GSUN	Golden Sun Health Technology Group Limited	2.871	-2.69%	5.69M	10.16M
288	CHNR	China Natural Resources, Inc.	0.575	-3.18%	5.67M	-
289	YGMZ	MingZhu Logistics Holdings Limited	0.780	-3.94%	5.21M	55.81M
290	ABTS	Abits Group Inc.	2.140	-14.06%	5.07M	5.34M
291	WLGS	WANG & LEE GROUP, Inc.	0.287	17.75%	4.99M	7.27M
292	SNTG	Sentage Holdings Inc.	1.710	-8.06%	4.80M	146.47K
293	WTO	UTime Limited	1.300	-8.45%	4.69M	32.31M
294	CPHI	China Pharma Holdings, Inc.	0.239	-4.48%	4.60M	4.53M
295	RTC	Baijiayun Group Ltd	0.226	-1.31%	4.35M	54.53M
296	TCTM	TCTM Kids IT Education Inc.	0.434	-5.66%	4.30M	190.00M
297	TWG	Top Wealth Group Holding Limited	0.145	-1.97%	4.19M	14.38M
298	JWEL	Jowell Global Ltd.	1.840	-	3.99M	160.01M
299	TAOP	Taoping Inc.	0.232	-7.79%	3.94M	42.64M
300	CNET	ZW Data Action Technologies Inc.	1.570	-1.26%	3.42M	18.46M
301	VSME	VS MEDIA Holdings Limited	0.918	0.63%	3.33M	8.49M
302	JXG	JX Luxventure Group Inc.	2.130	-6.99%	3.23M	38.49M
303	BAOS	Baosheng Media Group Holdings Limited	1.990	-4.33%	3.05M	859.23K
304	OST	Ostin Technology Group Co., Ltd.	1.449	-6.52%	3.02M	32.46M
305	IFBD	Infobird Co., Ltd	1.430	-3.38%	2.84M	300.00K
306	WNW	Meiwu Technology Company Limited	1.610	-4.17%	2.69M	380.35K
307	ITP	IT Tech Packaging, Inc.	0.260	-3.70%	2.62M	79.16M
308	KXIN	Kaixin Holdings	0.880	-4.36%	2.60M	12.68M
309	TANH	Tantech Holdings Ltd	1.920	-3.52%	2.28M	49.10M
310	LICN	Lichen International Limited	3.650	-12.26%	2.26M	41.93M
311	FAMI	Farmmi, Inc.	1.690	0.60%	2.11M	64.13M
312	EZGO	EZGO Technologies Ltd.	0.360	4.90%	2.04M	21.13M
313	TC	Token Cat Limited	0.678	-5.88%	2.03M	6.74M
314	DXF	Eason Technology Limited	6.59	-4.35%	1.64M	-56.60M
315	UK	Ucommune International Ltd	1.090	1.87%	1.57M	42.47M
316	CNEY	CN Energy Group. Inc.	0.125	-8.28%	1.53M	50.96M
317	AGMH	AGM Group Holdings Inc.	0.0470	-5.62%	1.14M	65.00M
318	NIVF	NewGenIvf Group Limited	0.561	-4.75%	952.86K	5.68M
319	BQ	Boqii Holding Limited	2.040	-3.77%	935.69K	81.21M
320	BON	Bon Natural Life Limited	0.0633	-11.96%	264.00K	23.84M
Back to Top ↑
Stock Analysis Pro
Upgrade now for unlimited access to all data and tools.
Market Newsletter
Get a daily email with the top market news in bullet point format.
Stock Screener
Filter, sort and analyze all stocks to find your next investment.
Watchlists
Keep track of your favorite stocks in real-time.
Sections
Stocks
IPOs
ETFs
Blog
Services
Stock Analysis Pro
Free Newsletter
Get Support
Website
Login
FAQ
Changelog
Sitemap
Advertise
Company
About
Contact Us
Terms of Use
Privacy Policy
Data Disclaimer
Affiliate Program
Market Newsletter
Daily market news in bullet point format.

Enter your email
Subscribe
© 2025 StockAnalysis.com. All rights reserved.
"""

def get_china_us_listed_tickers():
    """
    Get list of China stock tickers listed on US exchanges from the provided text data.
    """
    print("Parsing China US-Listed tickers from text data...")
    try:
        # Regex to find lines starting with a number, followed by tab/spaces,
        # then the ticker (uppercase letters), then tab/spaces, then the company name.
        # Captures the ticker symbol (group 1).
        pattern = re.compile(r"^\d+\s+([A-Z]+)\s+.*", re.MULTILINE)
        tickers = pattern.findall(CHINA_STOCK_LIST_TEXT)

        # Clean tickers (ensure uniqueness and uppercase, though regex should handle uppercase)
        tickers = sorted(list(set([ticker.strip() for ticker in tickers])))

        print(f"Found {len(tickers)} unique China US-Listed tickers")
        return tickers
    except Exception as e:
        print(f"Error parsing China US-Listed tickers: {e}")
        return []

def get_stock_data(ticker):
    """
    Get financial data for a single ticker using yfinance.
    Handles potential errors gracefully.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info # Fetch info first (less prone to empty data errors)

        # Basic check if info is valid
        if not info or info.get('regularMarketPrice') is None: # Check for a common field
            print(f"Warning: Could not retrieve valid info for {ticker}. Skipping.")
            return None

        # Get financials (might be empty or raise errors for some tickers)
        try:
            income_stmt = stock.income_stmt
        except Exception as e:
            print(f"Warning: Could not retrieve income statement for {ticker}: {e}. Proceeding without it.")
            income_stmt = pd.DataFrame() # Use empty DataFrame

        try:
            balance_sheet = stock.balance_sheet
        except Exception as e:
            print(f"Warning: Could not retrieve balance sheet for {ticker}: {e}. Proceeding without it.")
            balance_sheet = pd.DataFrame() # Use empty DataFrame

        # cash_flow = stock.cashflow # Often empty, removed for simplicity unless needed

        # Calculate metrics only if data is available
        revenue_growth = None
        earnings_growth = None
        debt_to_ebitda = None

        if not income_stmt.empty and 'Total Revenue' in income_stmt.index and len(income_stmt.loc['Total Revenue']) > 1:
            latest_revenue = income_stmt.loc['Total Revenue'].iloc[0]
            prev_revenue = income_stmt.loc['Total Revenue'].iloc[1]
            if prev_revenue and prev_revenue != 0: # Avoid division by zero
                revenue_growth = (latest_revenue - prev_revenue) / abs(prev_revenue) # Use abs for safety

        if not income_stmt.empty and 'Net Income' in income_stmt.index and len(income_stmt.loc['Net Income']) > 1:
            latest_earnings = income_stmt.loc['Net Income'].iloc[0]
            prev_earnings = income_stmt.loc['Net Income'].iloc[1]
            if prev_earnings and prev_earnings != 0: # Avoid division by zero
                earnings_growth = (latest_earnings - prev_earnings) / abs(prev_earnings) # Use abs for safety

        if not balance_sheet.empty and not income_stmt.empty and 'Total Debt' in balance_sheet.index and 'EBITDA' in income_stmt.index:
             # Ensure index alignment or use latest values carefully
            try:
                 total_debt = balance_sheet.loc['Total Debt'].iloc[0]
                 ebitda = income_stmt.loc['EBITDA'].iloc[0]
                 if total_debt is not None and ebitda is not None and ebitda != 0:
                     debt_to_ebitda = total_debt / ebitda
                 elif ebitda == 0 and total_debt is not None and total_debt > 0:
                     debt_to_ebitda = float('inf') # Indicate high debt if EBITDA is zero
                 else:
                     debt_to_ebitda = None # If either value is missing
            except IndexError:
                 print(f"Warning: IndexError accessing Debt or EBITDA for {ticker}.")
                 debt_to_ebitda = None
            except KeyError:
                 print(f"Warning: KeyError accessing Debt or EBITDA for {ticker}.")
                 debt_to_ebitda = None


        # P/E Ratio - Use forwardPE preferably, fallback to trailingPE if necessary
        pe_ratio = info.get('forwardPE')
        if pe_ratio is None:
            pe_ratio = info.get('trailingPE')

        # Store data
        return {
            'name': info.get('shortName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'), # Added industry
            'market_cap': info.get('marketCap', None), # Added market cap
            'revenue_growth': revenue_growth,
            'earnings_growth': earnings_growth,
            'pe_ratio': pe_ratio,
            'debt_to_ebitda': debt_to_ebitda,
            'price': info.get('regularMarketPrice'), # Use the checked value
            'volume': info.get('regularMarketVolume', None)
        }

    except Exception as e:
        # Catch broader exceptions during yfinance interaction
        print(f"Error processing {ticker} with yfinance: {e}")
        return None

def grade_stocks(stock_data, economic_quadrant):
    """
    Grade stock based on financial metrics and economic quadrant.
    (Using the same logic as S&P 500 for now, driven by quadrant letter)
    """
    # Create a copy of the stock data dictionary to avoid modifying the original
    graded_stock = stock_data.copy()

    # Default to grade C
    revenue_grade = 'C'
    earnings_grade = 'C'
    pe_grade = 'C'
    debt_grade = 'C'

    # Revenue Growth Grade (A, B, C, D, F)
    revenue_growth = stock_data.get('revenue_growth')
    if revenue_growth is not None:
        if revenue_growth > 0.25:  # >25% - Adjusted threshold slightly for growth focus
            revenue_grade = 'A'
        elif revenue_growth > 0.15:  # >15%
            revenue_grade = 'B'
        elif revenue_growth > 0.05:  # >5%
            revenue_grade = 'C'
        elif revenue_growth >= 0:  # >=0%
            revenue_grade = 'D'
        else:  # negative
            revenue_grade = 'F'

    # Earnings Growth Grade (A, B, C, D, F)
    earnings_growth = stock_data.get('earnings_growth')
    if earnings_growth is not None:
        if earnings_growth > 0.25:  # >25% - Adjusted threshold slightly
            earnings_grade = 'A'
        elif earnings_growth > 0.15:  # >15%
            earnings_grade = 'B'
        elif earnings_growth > 0.05:  # >5%
            earnings_grade = 'C'
        elif earnings_growth >= 0:  # >=0%
            earnings_grade = 'D'
        else:  # negative
            earnings_grade = 'F'

    # Grade P/E Ratio (Lower P/E is better)
    pe_ratio = stock_data.get('pe_ratio')
    if pe_ratio is not None and pe_ratio > 0:
        if pe_ratio <= 12:      # <12
            pe_grade = 'A'
        elif pe_ratio <= 18:      # <18
            pe_grade = 'B'
        elif pe_ratio <= 25:      # <25
            pe_grade = 'C'
        elif pe_ratio <= 35:      # <35
            pe_grade = 'D'        # D if high but positive
        else:
            pe_grade = 'F'        # F if very high
    elif pe_ratio is not None and pe_ratio <= 0:
        pe_grade = 'F' # F if negative P/E

    # Grade Debt to EBITDA (Lower is better)
    debt_to_ebitda = stock_data.get('debt_to_ebitda')
    if debt_to_ebitda is not None:
        if debt_to_ebitda == float('inf'):
            debt_grade = 'F' # Assign F if debt exists but EBITDA is zero/negative
        elif debt_to_ebitda <= 1.5: # <1.5
            debt_grade = 'A'
        elif debt_to_ebitda <= 3.0: # <3.0
            debt_grade = 'B'
        elif debt_to_ebitda <= 5.0: # <5.0
            debt_grade = 'C'
        elif debt_to_ebitda <= 8.0: # <8.0 - Allow higher leverage for some China sectors?
            debt_grade = 'D'
        else: # > 8.0
            debt_grade = 'F'

    # Add grades to stock data
    graded_stock['revenue_grade'] = revenue_grade
    graded_stock['earnings_grade'] = earnings_grade
    graded_stock['pe_grade'] = pe_grade
    graded_stock['debt_grade'] = debt_grade

    # --- Overall Grade Calculation (Quadrant-based) ---
    # Map letter grades to scores (A=4, B=3, C=2, D=1, F=0)
    grade_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
    score = 0
    num_grades = 0

    # Weights for each factor (customize if needed)
    weights = {
        'revenue_grade': 0.30,
        'earnings_grade': 0.30,
        'pe_grade': 0.20,
        'debt_grade': 0.20,
    }

    factors = ['revenue_grade', 'earnings_grade', 'pe_grade', 'debt_grade']
    for factor in factors:
        grade = graded_stock.get(factor)
        if grade in grade_map:
            score += grade_map[grade] * weights[factor]
            num_grades += weights[factor] # Use weight sum for normalization

    # Normalize score (0-4 scale)
    normalized_score = score / num_grades if num_grades > 0 else 0

    # --- Adjust score based on Economic Quadrant ---
    # This part requires defining how each quadrant favors certain characteristics.
    # Example: Quadrant A (Weakest Momentum) might favor strong balance sheets (low debt).
    # Quadrant D (Strongest Momentum) might favor high growth.

    quadrant_adjustment = 0 # Default no adjustment
    if economic_quadrant == 'A':
        # In weak momentum, favor low debt (debt_grade A/B) and positive earnings (not F)
        if debt_grade in ['A', 'B']:
             quadrant_adjustment += 0.2
        if earnings_grade != 'F':
             quadrant_adjustment += 0.1
    elif economic_quadrant == 'B':
         # Below average momentum, maybe look for value (P/E A/B) and decent growth (Rev/Earn C or better)
        if pe_grade in ['A', 'B']:
             quadrant_adjustment += 0.15
        if revenue_grade not in ['D', 'F'] and earnings_grade not in ['D', 'F']:
             quadrant_adjustment += 0.15
    elif economic_quadrant == 'C':
         # Above average momentum, favor growth (Rev/Earn A/B)
        if revenue_grade in ['A', 'B']:
             quadrant_adjustment += 0.2
        if earnings_grade in ['A', 'B']:
            quadrant_adjustment += 0.1
    elif economic_quadrant == 'D':
        # Strongest momentum, heavily favor growth (Rev/Earn A/B)
        if revenue_grade in ['A', 'B']:
            quadrant_adjustment += 0.3
        if earnings_grade in ['A', 'B']:
            quadrant_adjustment += 0.2

    # Apply adjustment and ensure score stays within 0-4 bounds
    final_score = min(max(normalized_score + quadrant_adjustment, 0), 4)

    # --- Map final score back to Overall Grade ---
    if final_score >= 3.7:
        overall_grade = 'A+'
    elif final_score >= 3.2:
        overall_grade = 'A'
    elif final_score >= 2.7:
        overall_grade = 'B+'
    elif final_score >= 2.2:
        overall_grade = 'B'
    elif final_score >= 1.7:
        overall_grade = 'C+'
    elif final_score >= 1.2:
        overall_grade = 'C'
    else: # Below 1.2
        overall_grade = 'D' # Changed F to D for less harsh grading initially

    graded_stock['overall_grade'] = overall_grade
    graded_stock['calculated_score'] = round(final_score, 3) # Store the score for reference

    return graded_stock


def save_china_data_to_database(graded_stocks, batch_name="china_us_listed"):
    """
    Save graded China stocks to the China-specific database.
    """
    print(f"Saving {len(graded_stocks)} China stocks to database (Batch: {batch_name})...")
    saved_count = 0
    skipped_count = 0

    for symbol, stock_data in graded_stocks.items():
        if stock_data is None or stock_data.get('overall_grade') is None:
            print(f"Skipping save for {symbol}: No data or grade.")
            skipped_count += 1
            continue
        try:
            # Prepare notes
            notes = f"Score: {stock_data.get('calculated_score', 'N/A'):.3f}; RevG: {stock_data.get('revenue_grade')}; EarnG: {stock_data.get('earnings_grade')}; PE_G: {stock_data.get('pe_grade')}; DebtG: {stock_data.get('debt_grade')}"

            # Prepare JSON data
            json_data = {
                'revenue_grade': stock_data.get('revenue_grade'),
                'earnings_grade': stock_data.get('earnings_grade'),
                'pe_grade': stock_data.get('pe_grade'),
                'debt_grade': stock_data.get('debt_grade'),
                'calculated_score': stock_data.get('calculated_score'),
                'sector': stock_data.get('sector'),
                'industry': stock_data.get('industry'),
                'market_cap': stock_data.get('market_cap')
            }

            # Save using the imported China-specific function
            success = save_china_stock_analysis(
                symbol=symbol,
                name=stock_data.get('name'),
                sector=stock_data.get('sector'), # Sector saved in json too, but keep top-level
                grade=stock_data.get('overall_grade'),
                price=stock_data.get('price'),
                volume=stock_data.get('volume'),
                momentum_score=None,  # Not calculated
                value_score=None,  # Not calculated (P/E grade used instead)
                growth_score=None,  # Not calculated (Rev/Earn grades used instead)
                quality_score=None, # Not calculated (Debt grade used instead)
                revenue_growth=stock_data.get('revenue_growth'),
                earnings_growth=stock_data.get('earnings_growth'),
                pe_ratio=stock_data.get('pe_ratio'),
                debt_to_ebitda=stock_data.get('debt_to_ebitda'),
                batch_name=batch_name,
                quadrant=stock_data.get('quadrant'), # Pass the determined China quadrant
                notes=notes,
                json_data=json_data
            )
            if success:
                saved_count += 1
            else:
                skipped_count +=1 # If save function indicated failure

        except Exception as e:
            print(f"Error saving {symbol} to China database: {e}")
            skipped_count += 1

    print(f"Attempted to save {len(graded_stocks)} stocks: {saved_count} saved, {skipped_count} skipped/failed.")
    return saved_count


def analyze_china_stocks():
    """
    Main function to analyze China stocks listed on US exchanges.
    """
    start_time = time.time()
    print("Starting China US-Listed stock analysis...")

    # Determine current China economic quadrant
    # Fetch more history for better Z-score/percentile calculation in quadrant func
    end_date_dt = datetime.now()
    start_date_dt = end_date_dt - pd.Timedelta(days=15*365) # 15 years history
    start_date_str = start_date_dt.strftime('%Y-%m-%d')
    end_date_str = end_date_dt.strftime('%Y-%m-%d')

    print(f"Determining China economic quadrant ({start_date_str} to {end_date_str})...")
    try:
        # Call the China-specific function
        quadrant, details, _ = determine_chinese_economic_quadrant(start_date=start_date_str, end_date=end_date_str)
        print(f"Current China Economic Quadrant: {quadrant} ({details.get('quadrant_description', 'No description')})")
        print(f"CMS Percentile: {details.get('cms_percentile', 'N/A'):.1f}%" if details.get('cms_percentile') is not None else "CMS Percentile: N/A")
    except Exception as e:
        print(f"Error determining China economic quadrant: {e}. Defaulting to 'C'.")
        quadrant = 'C' # Default to C if quadrant determination fails

    # Get all China US-Listed tickers
    tickers = get_china_us_listed_tickers()
    if not tickers:
        print("Failed to retrieve China US-Listed tickers. Exiting.")
        return None

    # Process tickers in batches
    all_graded_stocks = {}
    total_processed = 0
    total_retrieved = 0

    num_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(tickers), BATCH_SIZE):
        batch_start_index = i
        batch_end_index = min(i + BATCH_SIZE, len(tickers))
        batch_tickers = tickers[batch_start_index:batch_end_index]
        current_batch_num = (batch_start_index // BATCH_SIZE) + 1

        print(f"\n--- Processing Batch {current_batch_num}/{num_batches} (Tickers {batch_start_index+1}-{batch_end_index}) ---")

        batch_stocks = {}
        batch_retrieved_count = 0
        for ticker in batch_tickers:
            print(f"Processing {ticker}...")
            stock_data = get_stock_data(ticker)
            if stock_data:
                batch_retrieved_count += 1
                # Add quadrant information
                stock_data['quadrant'] = quadrant
                # Grade the stock
                try:
                    graded_stock = grade_stocks(stock_data, quadrant)
                    batch_stocks[ticker] = graded_stock
                    print(f"  -> Graded {ticker} ({stock_data.get('name', 'N/A')}): {graded_stock.get('overall_grade', 'N/A')} (Score: {graded_stock.get('calculated_score', 'N/A'):.3f})")
                except Exception as e:
                     print(f"  -> Error grading {ticker}: {e}")
            else:
                 print(f"  -> Skipping {ticker} (No data retrieved)")

            # Small delay to potentially avoid API rate limits
            time.sleep(0.2) # Slightly increased delay

        # Save batch to database
        batch_name = f"china_us_{datetime.now().strftime('%Y%m%d')}_batch_{current_batch_num}"
        if batch_stocks:
             saved_in_batch = save_china_data_to_database(batch_stocks, batch_name=batch_name)
             total_processed += saved_in_batch # Count successfully saved stocks
        else:
             print("No stocks graded in this batch to save.")

        total_retrieved += batch_retrieved_count
        print(f"--- Batch {current_batch_num} Complete: Retrieved {batch_retrieved_count}, Graded {len(batch_stocks)} ---")

        # Add graded stocks from batch to overall results (optional, can consume memory)
        # all_graded_stocks.update(batch_stocks)


    # Calculate execution time
    execution_time = time.time() - start_time
    print(f"\n--- Analysis Complete ---")
    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"Tickers Found: {len(tickers)}")
    print(f"Data Retrieved For: {total_retrieved} tickers")
    print(f"Successfully Graded & Saved: {total_processed} stocks")

    # Get China database statistics
    try:
        db_stats = get_china_db_stats()
        print("\n--- China Database Statistics ---")
        print(f"Total stock analyses in DB: {db_stats.get('china_stock_analysis_count', 'N/A')}")
        print(f"Total quadrant analyses in DB: {db_stats.get('china_economic_quadrant_count', 'N/A')}")
        print(f"Analysis batches found: {', '.join(db_stats.get('china_batches', ['N/A']))}")

        grade_distribution = db_stats.get('china_grade_distribution', {})
        if grade_distribution:
            print("\nGrade Distribution (All Time):")
            # Sort grades for consistent output
            sorted_grades = sorted(grade_distribution.keys(), key=lambda g: ('+', 'A', 'B', 'C', 'D', 'F').index(g[0]) if g else 99)
            for grade in sorted_grades:
                 count = grade_distribution[grade]
                 print(f"  {grade}: {count} stocks")
        else:
             print("\nGrade Distribution: No grading data found in DB.")

    except Exception as e:
        print(f"\nError retrieving database statistics: {e}")

    # Return all graded stocks (might be large, consider returning None or summary)
    # return all_graded_stocks
    print("--- Script Finished ---")
    return None # Return None to avoid printing large dict

if __name__ == "__main__":
    # Run the analysis
    analyze_china_stocks()

    # Example: How to query top stocks from DB later if needed
    # from db_utils_china import get_china_stocks_by_grade
    # top_a_plus = get_china_stocks_by_grade('A+', limit=10)
    # print("\nTop A+ Stocks from DB:")
    # print(top_a_plus[['symbol', 'name', 'sector', 'price', 'analysis_date']])