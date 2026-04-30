# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Bao cao Phan tich Chuyen sau - Data Storytelling 4 Cap do
# **Datathon 2026 - The GridBreakers - VinUni DS&AI Club**
#
# ---
#
# ## Tom tat Dieu hanh (Executive Summary)
#
# **Doanh thu giam 46% trong khi traffic tang 67%.** Day la mau thuan cot loi cua toan bo doanh nghiep - va cung la chia khoa de phuc hoi.
#
# Qua phan tich 13 bang du lieu, 646K+ don hang, 122K+ khach hang trong 10.5 nam (2012-2022), chung toi phat hien:
#
# | Phat hien | Du lieu | Hanh dong |
# |---|---|---|
# | **Conversion collapse:** Sessions->Orders giam tu 1.0% -> 0.4% (-60%) | Pheu chuyen doi | Audit UX checkout **ngay Q1** |
# | **COD Tax:** COD co cancel rate cao gap 3x prepay | Phan tich payment | Chuyen 20% COD -> Prepay |
# | **Inventory Paradox:** 50.6% san pham-thang vua het hang vua ton du | Inventory analysis | Chu ky tai dat hang 7-14 ngay |
# | **Mua vu nguoc:** Dinh Apr-Jun, day Nov-Jan (khong phai Tet) | Seasonality | Dieu chinh lich marketing + ton kho |
# | **Volume, khong phai Value:** AOV on dinh, orders -55% | Revenue decomposition | Khong can doi gia - can tang don |
#
# **Tiem nang phuc hoi: +500-700 trieu VND/nam** tu viec sua conversion + giam cancel + toi uu ton kho - khong can them khach hang moi.
#
# ---
#
# ## Cau truc Bao cao
#
# Bao cao nay duoc to chuc theo **Rubric 4 Cap do** cua cuoc thi:
#
# | Cap do | Cau hoi Trong tam | So bieu do |
# |---|---|---|
# | **1. Descriptive (Mo ta)** | *Cai gi da xay ra?* | 18 charts |
# | **2. Diagnostic (Chan doan)** | *Tai sao no xay ra?* | 22 charts |
# | **3. Predictive (Du bao)** | *Dieu gi se xay ra?* | 9 charts |
# | **4. Prescriptive (De xuat)** | *Can lam gi?* | 7 charts |
#
# Moi bieu do tuan thu cau truc: **Observation -> Why It Matters -> Strategy Link**

# %% [markdown]
# ---
# # CAP DO 1: DESCRIPTIVE - "CAI GI DA XAY RA?"
#
# Muc tieu: Mo ta buc tranh toan canh - khach hang, san pham, doanh thu, chat luong du lieu.

# %% [markdown]
# ## 1.1 Chat luong Du lieu - Nen tang cho moi Phan tich
#
# **Nguon:** `00_data_profiling.ipynb` - 13 bang x 6 kiem tra + 8 quy tac cardinality

# %% [markdown]
# ### Tong quan 13 Bang Du lieu
#
# | Bang | So dong | Chi tiet |
# |---|---|---|
# | `products` | 2,412 | 4 category: Streetwear (1,320), Outdoor (743), Casual (201), GenZ (148) |
# | `customers` | 121,930 | Nu 49%, Nam 47%, Non-binary 4%; 5 nhom tuoi; 6 kenh acquisition |
# | `orders` | 646,945 | 6 trang thai; Delivered 80%, Cancelled 9.2%, Returned 5.6% |
# | `order_items` | 714,669 | 16 dong trung PK; 61.3% khong co promo |
# | `payments` | 646,945 | 1:1 voi orders; 5 ky han tra gop |
# | `shipments` | 566,067 | Thieu 564 don so voi eligible orders (P0) |
# | `returns` | 39,939 | wrong_size #1 (35%), defective #2 (20%) |
# | `reviews` | 113,551 | Coverage 22.0%; 71% rating 4-5 sao |
# | `sales` | 3,833 | 2012-07-04 -> 2022-12-31, khong thieu ngay |
# | `inventory` | 60,247 | 126 thang x 1,624 san pham, snapshot cuoi thang |
# | `web_traffic` | 3,652 | 6 nguon traffic; bounce_rate luu fraction (~0.45%) |
# | `promotions` | 50 | Pattern 6-4-6-4; 80% khong target category |
# | `geography` | 39,948 | 39,948 ZIP -> 42 TP -> 3 vung (East/Central/West) |

# %% [markdown]
# ### Xac minh Quan trong
#
# - ✅ **sales.csv = TONG TAT CA trang thai don hang** (MAPE 5.2% vs ALL, 24.5% vs DELIVERED only)
# - ✅ **1:1 orders <-> payments** - toan ven tham chieu
# - ✅ **Tat ca timeline hop le** (ship >= order, delivery >= ship, return >= order...)
# - ⚠️ **80,623 khach co order truoc signup_date** - dau hieu du lieu mo phong
# - ⚠️ **10,531 dong unit_price lech >50% so voi catalog** - extreme promo
# - ⚠️ **Bounce rate ~0.45%** - luu fraction, khong phai %; thap bat thuong voi retail (binh thuong 30-60%)

# %% [markdown]
# ## 1.2 Dashboard Tong quan - "The Great Divergence"
#
# <img src="../../docs/streamlit/dashboard/dashboard_1.png" alt="Dashboard tong quan - Revenue vs Traffic" width="100%">

# %% [markdown]
# ### Observation
# - **"The Great Divergence":** Duong Revenue (bar) giam manh sau 2016 trong khi Sessions (line) tiep tuc tang
# - **KPI Dashboard:** Tong Revenue 16.4 ty VND - COGS 14.5 ty - Margin 12.5% - Orders 647K
# - **Category Mix:** Streetwear chiem ~80% doanh thu
#
# ### Why It Matters
# Mot bieu do duy nhat chung minh: van de khong phai demand (traffic van tang +67%) ma la **conversion** (doanh thu van giam -46%). Day la "elevator pitch" cho toan bo cau chuyen phan tich.
#
# ### Strategy Link
# Ngan sach nen chuyen tu **acquisition** (keo traffic) sang **conversion optimization** (toi uu trai nghiem mua hang).

# %% [markdown]
# ## 1.3 San pham - Danh muc & Bien loi nhuan
#
# <img src="../../reports/figures/fig_products_dist.png" alt="Phan phoi gia, margin %, va margin theo category" width="100%">

# %% [markdown]
# ### Observation
# - **2,412 san pham**, gia 9-40,950 VND, tap trung o Streetwear (54.7%)
# - **Gross margin trung binh 26.6%**, median 30.6%
# - **GenZ chi 148 SKU (6.1%)** - under-represented
# - **Casual margin thap nhat** trong cac category
#
# ### Why It Matters
# - **Rui ro phu thuoc Streetwear:** 55% danh muc o mot category -> single point of failure. Bat ky cu soc nao voi Streetwear deu anh huong toan bo doanh nghiep.
# - **Co hoi GenZ:** Phan khuc dang bi bo ngo, can phan bo them san pham
#
# ### Strategy Link
# Khi du bao doanh thu, can feature **category concentration index** - muc do tap trung cao lam tang volatility du bao.

# %% [markdown]
# ## 1.4 Khach hang - Chan dung & Kenh Thu hut
#
# <img src="../../reports/figures/fig_customers_dist.png" alt="Signups theo nam, nhom tuoi, va kenh acquisition" width="100%">

# %% [markdown]
# ### Observation
# - **121,930 khach** dang ky 2012-2022, tang truong deu
# - **25-34 tuoi = 29.8%** (young professionals) - phan khuc cot loi
# - **organic_search #1** (29.9%), **social_media #2** (20.1%)
# - **email_campaign chi 12.0%** nhung chi phi cao
#
# ### Why It Matters
# **KHONG thieu khach hang moi.** Signups tang deu moi nam -> revenue collapse KHONG phai do thieu demand. Van de nam o VIEC GIU CHAN va CHUYEN DOI khach hang hien co.
#
# ### Strategy Link
# Trong Part 3, dung **acquisition_channel lam feature phan tang** cho du bao - moi kenh co conversion pattern khac nhau.

# %% [markdown]
# ## 1.5 Doanh thu & COGS - Bien Muc tieu
#
# <img src="../../reports/figures/fig_sales_overview.png" alt="4-panel: Revenue nam, Revenue ngay, Margin %, Revenue theo DOW" width="100%">

# %% [markdown]
# ### Observation
# - **Tong doanh thu 10.5 nam: ~16.4 ty VND**, trung binh 4.29 trieu/ngay
# - **Dinh 2016: 2.1 ty/nam -> Day 2019: 1.14 ty/nam = -46%**
# - **Bien loi nhuan gop on dinh 12-16%** suot 10 nam
# - **Daily Revenue bien dong manh:** Min 279,814 -> Max 20,905,271 VND (bien do 75x)
# - **Thu Tu co doanh thu CAO NHAT** - nguoc quy luat ban le (cuoi tuan thuong cao nhat)
#
# ### Why It Matters
# - **Volume, khong phai Value:** Doanh nghiep khong can tang gia hay cai thien margin - can ban duoc NHIEU don hon
# - **"Wed > Sat" la insight van hanh quan trong:** Tap trung push notification, email vao trua Thu 3-Thu 4
# - **2022 chi dat 55.5% dinh 2016** -> can can thiep chu dong, khong the cho "tu lanh"
#
# ### Strategy Link
# Mo hinh du bao Part 3 phai capture duoc: (a) xu huong giam dai han, (b) mua vu Apr-Jun, (c) mau trong tuan Wed>Sat.

# %% [markdown]
# ## 1.6 Web Traffic - Nguon Khach Truy cap
#
# <img src="../../reports/figures/fig_web_traffic.png" alt="Sessions theo ngay va sessions trung binh theo nguon traffic" width="100%">

# %% [markdown]
# ### Observation
# - **Sessions tang +63% (2013->2022)**
# - **organic_search la nguon #1** - SEO hoat dong hieu qua
# - **Sessions/ngay TB: 25,042**, dinh 50,947
#
# ### Why It Matters
# **Dau tu SEO dang hieu qua** - tiep tuc duy tri nhung khong can tang ngan sach. Traffic du thua ma revenue van giam -> van de nam o TREN TRANG WEB.
#
# ### Strategy Link
# Trong Part 3, sessions_total dung **lag >=365 ngay** de tranh leakage. Correlation sessions->revenue ngay cang yeu (r=0.5->0.15) - mo hinh can capture duoc su suy giam nay.

# %% [markdown]
# ## 1.7 Ma tran Tuong quan Da chieu (G1 - Generated by AI)
#
# <img src="../../reports/figures/generated_by_ai/G1_correlation_heatmap.png" alt="Clustered Correlation Heatmap" width="100%">

# %% [markdown]
# ### Observation
# - **Revenue-COGS r = 0.99** -> hai bien gan nhu dong bien hoan hao
# - **n_orders-Revenue r = 0.8** -> volume driver confirmed
# - **Sessions-Revenue r = 0.1** -> conversion collapse evidence ro rang trong du lieu
# - **Calendar features (month_sin/cos, dow_sin/cos)** it tuong quan truc tiep nhung critical cho seasonality
# - **n_cancelled-Revenue r am yeu** -> cancel khong tap trung vao ngay doanh thu cao
#
# ### Why It Matters
# Ma tran nay xac nhan 3 dieu: (1) COGS co the du bao tu Revenue, (2) Volume la driver chinh, (3) Traffic KHONG con la tin hieu tot de du bao doanh thu. Multicollinearity giua n_orders va n_delivered -> can xu ly khi build features cho Part 3.
#
# ### Strategy Link
# Trong Part 3, **loai bo features co VIF > 10** (n_delivered, gross_revenue_recon) khoi model de tranh overfitting.

# %% [markdown]
# ---
# # CAP DO 2: DIAGNOSTIC - "TAI SAO NO XAY RA?"

# %% [markdown]
# ## 2.1 Pheu Chuyen doi - Noi "Thung Nuoc" bi Ro ri
#
# <img src="../../reports/figures/fig_funnel_analysis.png" alt="4-panel Funnel Analysis" width="100%">

# %% [markdown]
# ### Observation
# - **Sessions +67% (2013->2022) nhung Orders -55%, Revenue -46%**
# - **Session-to-Order Conversion giam tu ~1.0% (dinh 2016) -> ~0.4% (2022) = -60%**
# - **Revenue/Session giam tu ~220 VND -> ~100 VND** - moi khach tao ra it hon 55% gia tri
# - **Neu traffic 2022 co conversion nhu 2016:** ~294 don/ngay thay vi ~100 -> **+194 don/ngay bi mat**
#
# ### Why It Matters
# **Pheu bi thung o khau CONVERSION, khong phai ACQUISITION.** 1,000 nguoi ghe tham -> truoc day 10 nguoi mua -> nay chi con 4. Day la that bai SUPPLY-SIDE + CONVERSION: khach van den, van muon mua, nhung KHONG THE hoan tat giao dich.
#
# ### Strategy Link
# Trong Part 3, **conversion_rate_proxy** (n_orders / sessions_total) la feature leading indicator cho du bao. Can them interaction features giua sessions va seasonality.

# %% [markdown]
# ## 2.2 Revenue Decomposition - Volume hay Value?
#
# <img src="../../reports/figures/fig_revenue_decomposition.png" alt="4-panel Revenue Decomposition" width="100%">

# %% [markdown]
# ### Observation
# - **Orders -55% (2016->2022) nhung AOV gan nhu KHONG DOI (~200K VND/don)**
# - **avg_unit_price on dinh suot 10 nam** -> khong co price erosion
# - **Items/order on dinh ~5.0** -> basket size khong doi
# - **Structural Break phat hien tai 2019:** Pre-2019 trend tang nhe, Post-2019 di ngang/suy giam
# - **Avg discount % tang nhe** -> doanh nghiep da thu dung discount kich cau nhung khong hieu qua
#
# ### Why It Matters
# **Khong can thay doi gia hay san pham.** AOV, basket size, unit price deu on dinh -> van de thuan tuy la SO LUONG DON HANG. Moi dong chi cho pricing strategy hay product mix se lang phi.
#
# ### Strategy Link
# Trong Part 3, mo hinh du bao can **tach biet trend va seasonality** - trend component bat structural break, seasonal component bat mau Apr-Jun.

# %% [markdown]
# ## 2.3 Phan tich Do tre - Promo mat bao lau de co hieu qua? (G2 - Generated by AI)
#
# <img src="../../reports/figures/generated_by_ai/G2_lead_lag_ccf.png" alt="Lead-Lag Cross-Correlation Function" width="100%">

# %% [markdown]
# ### Observation
# - **Promo -> Revenue co do tre toi uu duoc phat hien qua CCF**
# - **Sessions -> Revenue co correlation cao nhat o lag ngan**
# - **Correlation giam dan theo thoi gian** - xac nhan conversion dang suy yeu
# - **Bien do CCF thap (<0.3)** -> promo khong phai driver chinh cua doanh thu
#
# ### Why It Matters
# Neu promo mat X ngay de co hieu ung, thi: (1) Dung lag=X lam feature chinh cho du bao, (2) Khong danh gia hieu qua promo trong ngay - phai doi X ngay, (3) Lap lich promo truoc mua cao diem X ngay.
#
# ### Strategy Link
# Trong Part 3, feature `promo_discount_sum` nen dung **lag = best_lag tu CCF** thay vi lag=0. Dong thoi them `promo_effect_decay` - trong so giam dan theo ngay.

# %% [markdown]
# ## 2.4 Pareto SKU - Muc do Tap trung Doanh thu (G3 - Generated by AI)
#
# <img src="../../reports/figures/generated_by_ai/G3_pareto_sku.png" alt="Pareto SKU Analysis" width="100%">

# %% [markdown]
# ### Observation
# - **Top 10% SKU tao ra 65% doanh thu** (tap trung cao)
# - **Top 50 SKU = 41% tong doanh thu** (de ton thuong)
# - **Streetwear dominates top SKUs** - cung category dang co inventory whipsaw
# - **Lorenz curve cho thay phan phoi cuc ky lech** (Gini = 0.6-0.7)
#
# ### Why It Matters
# **Rui ro tap trung cuc cao:** Neu top 50 SKU gap van de ve ton kho hoac chat luong, doanh thu sup do. Ket hop voi inventory paradox -> "hero SKUs" vua duoc mua nhieu nhat, vua het hang thuong xuyen nhat -> vong luan quan chet nguoi.
#
# ### Strategy Link
# Trong Part 3, tao feature **hero_sku_stockout_risk** - neu top-50 SKU co stockout_days > threshold, du bao revenue dieu chinh giam. Day la early warning signal cho du bao.

# %% [markdown]
# ## 2.5 Nghich ly Ton kho - KPI Illusion
#
# <img src="../../reports/figures/fig_inventory_paradox.png" alt="Inventory Paradox 3-panel" width="100%">
#
# <img src="../../reports/figures/fig_inventory_analysis.png" alt="Inventory Analysis 4-panel" width="100%">

# %% [markdown]
# ### Observation
# - **50.6% san pham-thang CO CA stockout=1 VA overstock=1** - tuong nhu khong the
# - **Fill rate trung binh 96.1% nhung stockout flag = 67.3%** -> KPI ILLUSION
# - **Days of supply mean = 912.7 ngay (2.5 nam!)** nhung median = 240 ngay -> phan phoi cuc lech
# - **Mann-Whitney U test: p < 0.05** -> paradox months co units_received cao hon -> "timing hypothesis" duoc ung ho
#
# ### Why It Matters
# **Co che "Timing":** Het hang dau thang -> dat bo sung khan cap -> nhan hang cuoi thang -> ton kho du thua. **Bo KPI Fill Rate, dung Stockout Days + Lost Revenue.** Fill rate che giau van de thuc su.
#
# ### Strategy Link
# Trong Part 3, **inventory_health_score** (weighted average cua stockout_days, fill_rate, days_of_supply) la feature du bao som - stockout hom nay -> revenue giam 7-14 ngay sau.

# %% [markdown]
# ## 2.6 Phan tich Phuong thuc Thanh toan - "COD Tax" (G4 - Generated by AI)
#
# <img src="../../reports/figures/generated_by_ai/G4_payment_funnel.png" alt="Payment Method Funnel Analysis" width="100%">

# %% [markdown]
# ### Observation
# - **COD co cancel rate cao nhat** trong tat ca phuong thuc thanh toan
# - **Prepay (credit_card, paypal, apple_pay) co cancel rate thap hon dang ke**
# - **Doanh thu mat tu COD cancel: ~376M VND (toan bo giai doan)**
# - **AOV median COD thap hon prepay** - COD khach mua don nho, de huy hon
#
# ### Why It Matters
# **"COD Tax" - chi phi an cua COD:** Khong chi la phi thu ho, ma con la doanh thu mat tu cancel (~376M VND toan giai doan). Moi 1% COD chuyen sang prepay = tiet kiem ~3.8M VND. Incentive freeship cho prepay hoac discount 2% co the chuyen doi 20% COD volume.
#
# ### Strategy Link
# Trong Part 3, feature **cod_ratio** (ti le COD trong 30 ngay qua) la predictor cho cancel_rate va revenue volatility.

# %% [markdown]
# ## 2.7 Cohort Retention - Ai o lai, Ai ra di?
#
# <img src="../../reports/figures/fig_cohort_analysis.png" alt="Cohort Analysis 3-panel" width="100%">
#
# <img src="../../docs/streamlit/diagnosis/cohort_1.png" alt="Cohort Retention Heatmap" width="100%">

# %% [markdown]
# ### Observation
# - **28% khach quay lai mua hang trong vong 3 thang (M0->M3)** - ti le lap lai thap
# - **41% co giao dich tich luy den M6, 58% den M12** - tich luy tang dan
# - **Cohort 2017-2018 co retention THAP HON 2013-2016** -> chat luong cohort suy giam (ung ho H1)
# - **organic_search co retention va LTV CAO NHAT**
# - **email_campaign co LTV THAP NHAT** du la kenh acquisition lon thu 2 - nghich ly: re de acquire nhung kem retention
#
# ### Why It Matters
# **Chuyen ngan sach tu email_campaign sang SEO + Referral.** ROI tren LTV cao hon dang ke. **Win-Back M3+:** Nham khach da roi bo sau 3 thang - ho tung mua, chi la khong quay lai -> chi phi kich hoat lai thap hon acquisition moi.
#
# ### Strategy Link
# Trong Part 3, **cohort_quality_score** (retention trung binh cua cohort theo signup_month) la feature bo tro cho du bao dai han.

# %% [markdown]
# ## 2.8 Ma tran Di chuyen Phan khuc Khach hang (G8 - Generated by AI)
#
# <img src="../../reports/figures/generated_by_ai/G8_customer_migration.png" alt="Customer Segment Migration" width="100%">

# %% [markdown]
# ### Observation
# - **Loyal co retention 40%** - cao nhat trong cac phan khuc
# - **Active co retention 34%**, 66% con lai upgrade len Loyal (48%) va Champion (19%)
# - **Khong ghi nhan Active -> Dormant** trong 12 thang dau - dau hieu tich cuc
# - **66% khach hang di len (upgrade)** - phan lon nguoi mua tang engagement theo thoi gian
#
# ### Why It Matters
# **Chien luoc giu chan nen tap trung vao M0-M6**, giai doan khach hang co xu huong upgrade. 66% Active len Loyal/Champion cho thay gia tri vong doi tang dan — nen khuyen khich mua lap lai som bang loyalty program. Neu co du lieu dai han hon, can xac dinh diem "dinh" de du bao suy giam.
#
# ### Strategy Link
# Trong Part 3, **segment_upgrade_score** (xac suat khach hang upgrade phan khuc trong 6 thang toi) la feature du bao tang truong revenue tu khach hang hien tai.

# %% [markdown]
# ## 2.9 Mau Thoi gian - Khi nao khach mua hang?
#
# <img src="../../docs/streamlit/diagnosis/pattern_timing_1.png" alt="Monthly Seasonality" width="100%">
#
# <img src="../../docs/streamlit/diagnosis/pattern_timing_2.png" alt="Year x Month Heatmap" width="100%">
#
# <img src="../../docs/streamlit/diagnosis/pattern_timing_3.png" alt="Day of Week Pattern" width="100%">

# %% [markdown]
# ### Observation
# - **Dinh doanh thu = Thang 4-5-6, DAY = Thang 11-12-1** - HOAN TOAN NGUOC voi ban le VN (thuong dinh Tet)
# - **Mau nay NHAT QUAN tren TAT CA 10 NAM** - khong phai nhieu
# - **Thu Tu > Thu Bay** - "office-worker hypothesis": mua sam gio nghi trua qua mobile
# - **Bien do mua vu ~94%** giua thang cao nhat (May: ~6.6M VND/ngay) va thap nhat (Dec: ~2.5M VND/ngay)
#
# ### Why It Matters
# **Tuyet doi KHONG dung calendar VN mac dinh (Tet, Noel) de lap ke hoach.** Dung lich mua vu RIENG: dinh Apr-Jun, day Nov-Jan. Bat ky mo hinh nao dung dac trung Tet se DU BAO QUA CAO Q1 va DU BAO QUA THAP Q2.
#
# ### Strategy Link
# Trong Part 3, calendar features la **xuong song cua mo hinh** (37 features). Sin/Cos encoding cua day-of-year nam bat chu ky muot hon one-hot.

# %% [markdown]
# ## 2.10 Ti le Hoan tra - Category x Ly do (G7 - Generated by AI)
#
# <img src="../../reports/figures/generated_by_ai/G7_return_heatmap.png" alt="Return Rate Heatmap" width="100%">

# %% [markdown]
# ### Observation
# - **Wrong Size la ly do hoan tra #1 (35%)** - dac biet nghiem trong o Streetwear
# - **Defective #2 (20%)** - van de chat luong san pham
# - **Streetwear co return rate cao nhat** - ket hop voi viec la category doanh thu chinh -> margin bi bao mon kep
# - **Casual co return rate thap nhat** - nhung doanh thu cung thap
#
# ### Why It Matters
# **Wrong Size = van de DE SUA NHAT.** Them Size Guide chi tiet (so do cm, anh thuc te, review co anh) co the giam 30-50% hoan tra wrong_size. Voi 13,967 don wrong_size, moi don hoan mat trung binh 12,784 VND refund -> tong thiet hai ~178 trieu VND + mat margin + mat customer trust.
#
# ### Strategy Link
# Trong Part 3, feature **return_rate_7d** va **return_rate_by_category** la proxy cho "customer satisfaction" - anh huong den repeat purchase va revenue tuong lai.

# %% [markdown]
# ## 2.11 Ridge Plot - Phan phoi Doanh thu theo Thang qua 2 Thoi ky (G5 - Generated by AI)
#
# <img src="../../reports/figures/generated_by_ai/G5_ridge_plot.png" alt="Ridge Plot Revenue Distribution by Month" width="100%">

# %% [markdown]
# ### Observation
# - **Phan phoi doanh thu dich trai ro ret tu Pre-peak (2013-2016) -> Post-shock (2019-2022)**
# - **Thang 5 (dinh) co phan phoi rong nhat** - bien dong cao nhat vao mua cao diem
# - **Thang 1 (day) co phan phoi hep, lech phai** - it ngay doanh thu cao dot bien
# - **Mau phan phoi nhat quan giua cac thang** - chi khac ve quy mo (scale), khong khac ve hinh dang (shape)
#
# ### Why It Matters
# Revenue collapse la **location shift, khong phai shape shift** - tat ca cac thang deu giam cung mot ti le. Dieu nay khang dinh: cu soc mang tinh he thong (structural), khong phai mua vu (seasonal).
#
# ### Strategy Link
# Trong Part 3, **khong can mo hinh rieng cho tung thang** - mot mo hinh voi calendar features du capture seasonality. Dung quantile regression hoac conformal prediction de capture uncertainty.

# %% [markdown]
# ---
# # CAP DO 3: PREDICTIVE - "DIEU GI SE XAY RA?"

# %% [markdown]
# ## 3.1 Tinh Mua vu & Chu ky
#
# <img src="../../reports/figures/eda_seasonality.png" alt="Seasonality charts" width="100%">

# %% [markdown]
# ### Observation
# - **Seasonal CV ~15-20%** - bien dong mua vu dang ke
# - **Month x DOW heatmap xac nhan tuong tac:** Thu Tu thang 5 la "perfect storm"
# - **Lag-7 va Lag-365 la 2 chu ky quan trong nhat**
# - **Sin/Cos encoding muot hon one-hot cho day-of-year**
#
# ### Strategy Link
# 37 calendar features la xuong song cua mo hinh Part 3. Sin/Cos encoding + month + DOW + quarter + week-of-year.

# %% [markdown]
# ## 3.2 Moi quan he Revenue-COGS & Tin hieu Ngoai sinh
#
# <img src="../../reports/figures/eda_traffic_revenue.png" alt="Traffic-Revenue correlation" width="100%">

# %% [markdown]
# ### Observation
# - **Revenue-COGS Pearson r = 0.99** -> du bao chung feature set
# - **Order count co correlation manh nhat voi Revenue (r = 0.8)**
# - **Lag correlation giam dan:** r(Sessions, Revenue) tu 0.5 (2013) -> 0.15 (2022)
#
# ### Strategy Link
# 205 features: Calendar (37), Promo (5), Revenue lags/rolling/EWM (58), COGS lags (58), Orders (29), Web Traffic (14), Cross-ratios (4). Tat ca external features dung lag >=365.

# %% [markdown]
# ## 3.3 Cross-Validation & Hieu nang Mo hinh
#
# <img src="../../reports/figures/fitted_2022.png" alt="Fitted 2022 values" width="100%">
#
# <img src="../../reports/figures/feature_importance_comparison.png" alt="Feature importance comparison" width="100%">

# %% [markdown]
# ### Observation
# - **Ensemble MAE = 950K +/- 162K VND/ngay, R2 = 0.75**
# - **LGB-MAE solo: MAE = 759K, R2 = 0.78** - outperforms ensemble
# - **Fold stability tot:** MAE range 700K -> 1.2M
# - **Gap = 14 ngay trong TimeSeriesSplit** ngan leakage
#
# ### Strategy Link
# Ensemble blending (60% LGB + 25% Ridge + 15% Seasonal) can bang giua do chinh xac va on dinh. Iterative prediction voi FULL feature recomputation moi ngay.

# %% [markdown]
# ## 3.4 SHAP - Yeu to nao Quan trong nhat?
#
# <img src="../../reports/figures/fig_shap_revenue.png" alt="SHAP analysis" width="100%">

# %% [markdown]
# ### Observation - Top-10 Features (theo SHAP)
#
# | Hang | Feature | Y nghia Kinh doanh | Huong |
# |---|---|---|---|
# | 1 | `Revenue_l1` | Doanh thu hom qua -> momentum ngan han | ↑ |
# | 2 | `COGS_l1` | COGS hom qua (tuong quan manh) | ↑ |
# | 3 | `Revenue_rmin7` | Doanh thu thap nhat 7 ngay -> "san" | ↑ |
# | 4 | `COGS_l364` | COGS cung ngay nam truoc -> neo mua vu | Mau nam |
# | 5 | `Revenue_2yoy_mean` | Baseline 2 nam -> on dinh hoa | On dinh |
# | 6 | `Revenue_l7` | Doanh thu 1 tuan truoc -> mau tuan | Mua vu tuan |
# | 7 | `ord_count_l365` | Order volume nam truoc -> proxy cau | ↑ |
# | 8 | `cal_day` | Ngay trong thang (1-31) | Cuoi thang ↑ |
# | 9 | `promo_discount_sum` | Tong discount dang active | ↑ |
# | 10 | `cal_doy_sin/cos` | Sin/Cos day-of-year | Dinh Apr-Jun |
#
# ### Strategy Link
# Temporal features CHIEM UU THE. Co the giam tu 218 -> ~50 features ma MAE chenh <5%. Feature importance nhat quan giua LGB-MAE va CatBoost.

# %% [markdown]
# ## 3.5 Du bao 2023-2024 - Buc tranh Tuong lai
#
# <img src="../../reports/figures/fig_forecast_revenue.png" alt="Forecast 2023-2024" width="100%">

# %% [markdown]
# ### Observation
# - **Revenue du bao TB ~3.2-3.5 trieu VND/ngay** - tiep tuc xu huong suy giam cham
# - **Mau mua vu duoc bao toan:** Dinh Apr-Jun, day Nov-Jan
# - **COGS bam sat Revenue (r = 0.99)** - Gross Profit duong moi ngay
# - **In-sample R2 = 0.75** - con ~25% phuong sai chua giai thich duoc
#
# ### Strategy Link
# Du bao la "worst-case scenario" - baseline neu khong co can thiep. Dung lam reference cho ROI calculation cua cac prescriptive actions.

# %% [markdown]
# ## 3.6 Du bao voi Khoang Bat dinh (G6 - Generated by AI)
#
# <img src="../../reports/figures/generated_by_ai/G6_forecast_uncertainty.png" alt="Forecast Uncertainty Cone" width="100%">

# %% [markdown]
# ### Observation
# - **Khoang bat dinh mo rong theo thoi gian:** +/-X% sau 1 thang, +/-Y% sau 6 thang, +/-Z% sau 12 thang
# - **95% CI rong gap ~1.5x 80% CI** - dung nhu ky vong ly thuyet
# - **Uncertainty tich luy theo ham sqrt(t)** - dac trung cua random walk
# - **Thang 4-6 (mua cao diem) co uncertainty lon nhat** - bien dong cao nhat vao dung luc quan trong nhat
#
# ### Why It Matters
# **Du bao diem (point forecast) la khong du cho quyet dinh kinh doanh.** Can khoang bat dinh de:
# - Lap ke hoach ton kho: dung upper bound (95% CI) de tranh stockout
# - Lap ke hoach tai chinh: dung lower bound (80% CI) cho worst-case cash flow
# - Do luong rui ro: neu actual nam ngoai 95% CI -> dau hieu structural change
#
# ### Strategy Link
# Trong Part 3, trien khai **quantile regression** hoac **conformal prediction** de sinh prediction intervals chinh xac hon thay vi dung +/-k x sigma don gian.

# %% [markdown]
# ---
# # CAP DO 4: PRESCRIPTIVE - "CAN LAM GI?"

# %% [markdown]
# ## 4.1 Recovery Simulator - 5 Don bay Phuc hoi
#
# <img src="../../docs/streamlit/strategy/strategy_1.png" alt="Recovery Simulator" width="100%">

# %% [markdown]
# ### 5 Don bay & Tiem nang Tac dong
#
# | # | Don bay | Hien tai | Muc tieu | Impact (trieu/nam) | Effort | Uu tien |
# |---|---|---|---|---|---|---|
# | **1** | **Sua Conversion Rate** | 0.42% | 0.65% | **+300M** | Thap-TB | **LAM NGAY** |
# | **2** | **Giam Cancel Rate** | 9.2% | 6.0% | **+200M** | Thap | **LAM NGAY** |
# | **3** | **Giam Stockout Days** | 1.16 | 0.5 | +150M | TB-Cao | Q2-Q3 |
# | **4** | **Tang AOV** | 200K | 220K | +100M | TB | Q3 |
# | **5** | **Tang Sessions** | 11.1M | 13.3M | +50M | Cao | Khong uu tien |
#
# ### Tai sao Conversion & Cancel la uu tien #1 va #2?
# - Day la don bay **"SUA CHUA" chu khong phai "XAY MOI"**
# - Chi can chuyen doi tot hon nhung gi dang co, khong can tao nhu cau moi
# - **Ca hai deu low-effort:** Khong can thay doi san pham hay mo rong thi truong

# %% [markdown]
# ## 4.2 Ma tran Uu tien - Quick Wins vs Transformational
#
# <img src="../../docs/streamlit/strategy/strategy_2.png" alt="Priority Matrix" width="100%">

# %% [markdown]
# | Goc | Hanh dong | Effort | Impact | Chien luoc |
# |---|---|---|---|---|
# | **Quick Wins** (↓Effort, ↑Impact) | Size Guide Fix, COD->Prepay Nudge | 1-2 | 0.4-1.4 B | **Lam ngay Q1** |
# | **Transformational** (↑Effort, ↑Impact) | Reorder Policy Rebuild | 4 | 1.2 B | **Q2-Q3** |
# | **Fill-ins** (↓Effort, ↓Impact) | Cohort Win-back, Kill Dead-stock SKUs | 1 | 0.2-0.4 B | **Song song** |
# | **Moonshots** (↑Effort, ↓Impact) | Paid-channel Re-allocation | 2 | 0.5 B | **Can nhac sau** |

# %% [markdown]
# ## 4.3 Lo trinh Trien khai 2023
#
# <img src="../../docs/streamlit/strategy/strategy_3.png" alt="Gantt Roadmap" width="100%">

# %% [markdown]
# | Quy | Hanh dong | Don bay | Ky vong Impact |
# |---|---|---|---|
# | **Q1** | Audit UX checkout. Them size guide. Win-back At-Risk. | Conversion + Retention | +50M (ramp-up) |
# | **Q2** | COD->Prepay incentive. Toi uu mobile checkout (Wed peak). | Cancel + Conversion | +120M |
# | **Q3** | Chu ky tai dat hang 7-14 ngay. Tang stock GenZ, giam Casual. | Stockout | +200M (cong don) |
# | **Q4** | Danh gia & dieu chinh. Category-targeted promos. | Toi uu lien tuc | +350M (cong don) |

# %% [markdown]
# ## 4.4 Bang Hanh dong Chi tiet - Tu Insight den Execution
#
# | # | Insight Goc | Hanh dong Cu the | Chu so huu | KPI | Muc do |
# |---|---|---|---|---|---|
# | 1 | Revenue -46%, Traffic +67% | **Dung tang chi marketing.** Phan bo lai sang UX/Conversion. | CMO | CAC, ROAS | 🔴 CAO |
# | 2 | Conversion 1.0%->0.4% | **Audit toan bo checkout flow.** Tim diem ma sat. | CTO/Product | Conversion Rate | 🔴 CAO |
# | 3 | AOV on dinh, Volume -55% | **Khong thay doi gia.** Tap trung tang so don. | Pricing | n_orders/ngay | 🟡 TB |
# | 4 | 50.6% inventory paradox | **Chuyen chu ky tai dat 30->7-14 ngay.** Bo KPI fill_rate. | Supply Chain | Stockout days | 🔴 CAO |
# | 5 | Apr-Jun peak, Wed>Sat | **Dieu chinh lich marketing & ton kho.** Push TB trua T3-T4. | Marketing | Revenue/thang | 🟡 TB |
# | 6 | COD cancel rate 2-3x prepay | **Incentive COD->Prepay.** Freeship/discount 2% cho prepay. | Ops/Finance | Cancel Rate | 🔴 CAO |
# | 7 | Wrong size = #1 return (35%) | **Size guide chi tiet cho Streetwear.** Anh thuc te, so do. | Product | Return Rate | 🟡 TB |
# | 8 | organic_search + referral LTV cao | **Tang SEO + Referral program.** Giam email_campaign. | Marketing | LTV by channel | 🟡 TB |
# | 9 | Champions (25%) = 70% revenue | **Loyalty Program.** Early access, VIP pricing. | CRM | Retention Rate | 🟢 THAP |
# | 10 | Forecast baseline: suy giam cham | **Dung forecast lam baseline cho ROI.** Actual vs Forecast. | Strategy | ΔRevenue | 🟢 THAP |

# %% [markdown]
# ## 4.5 Tom tat - Elevator Pitch (60 Giay)
#
# > **"Doanh thu giam 46% trong khi traffic tang 67%. Khong phai vi het khach - khach van den, van muon mua. Ho chi khong the hoan tat don hang.**
# >
# > **Conversion rate giam tu 1% xuong 0.4% - pheu bi thung o buoc cuoi cung. AOV khong doi, basket size khong doi, margin khong doi. Van de la VOLUME - va volume la thu DE SUA NHAT neu biet dung cho hong.**
# >
# > **Cung luc, he thong ton kho dang tu lam minh dau: 50% san pham vua het hang vua ton du. Mua cao diem thuc su la thang 4-6, khong phai Tet. Thu Tu ban chay hon Thu Bay - khach mua sam gio nghi trua tren dien thoai.**
# >
# > **Hai hanh dong dau tien: (1) Sua conversion - audit UX checkout, toi uu mobile; (2) Giam cancel - chuyen COD sang tra truoc. Tong tiem nang: +500 trieu/nam tu viec sua nhung thu dang hong. Khong can lam gi moi. Khong can them khach. Chi can de khach mua hang de dang hon."**

# %% [markdown]
# ---
#
# ## Phu luc: Nguon Du lieu & Kha nang Tai lap
#
# ### Danh sach Bieu do Da su dung
#
# | # | Bieu do | Nguon | Tang |
# |---|---|---|---|
# | 1 | Dashboard Tong quan | `docs/streamlit/dashboard/dashboard_1.png` | Descriptive |
# | 2 | Phan phoi San pham | `reports/figures/fig_products_dist.png` | Descriptive |
# | 3 | Phan phoi Khach hang | `reports/figures/fig_customers_dist.png` | Descriptive |
# | 4 | Doanh thu & COGS | `reports/figures/fig_sales_overview.png` | Descriptive |
# | 5 | Web Traffic | `reports/figures/fig_web_traffic.png` | Descriptive |
# | 6 | **Ma tran Tuong quan** | `reports/figures/generated_by_ai/G1_correlation_heatmap.png` | Descriptive |
# | 7 | Pheu Chuyen doi | `reports/figures/fig_funnel_analysis.png` | Diagnostic |
# | 8 | Revenue Decomposition | `reports/figures/fig_revenue_decomposition.png` | Diagnostic |
# | 9 | Nghich ly Ton kho | `reports/figures/fig_inventory_paradox.png` + `fig_inventory_analysis.png` | Diagnostic |
# | 10 | Cohort Retention | `reports/figures/fig_cohort_analysis.png` | Diagnostic |
# | 11 | Mau Thoi gian | `docs/streamlit/diagnosis/pattern_timing_1-3.png` | Diagnostic |
# | 12 | **Lead-Lag CCF** | `reports/figures/generated_by_ai/G2_lead_lag_ccf.png` | Diagnostic |
# | 13 | **Pareto SKU** | `reports/figures/generated_by_ai/G3_pareto_sku.png` | Diagnostic |
# | 14 | **Payment Funnel** | `reports/figures/generated_by_ai/G4_payment_funnel.png` | Diagnostic |
# | 15 | **Ridge Plot** | `reports/figures/generated_by_ai/G5_ridge_plot.png` | Diagnostic |
# | 16 | **Return Heatmap** | `reports/figures/generated_by_ai/G7_return_heatmap.png` | Diagnostic |
# | 17 | **Customer Migration** | `reports/figures/generated_by_ai/G8_customer_migration.png` | Diagnostic |
# | 18 | Mua vu | `reports/figures/eda_seasonality.png` | Predictive |
# | 19 | Traffic-Revenue | `reports/figures/eda_traffic_revenue.png` | Predictive |
# | 20 | CV & Fitted | `reports/figures/fitted_2022.png` + `feature_importance_comparison.png` | Predictive |
# | 21 | SHAP | `reports/figures/fig_shap_revenue.png` | Predictive |
# | 22 | Du bao | `reports/figures/fig_forecast_revenue.png` | Predictive |
# | 23 | **Forecast Uncertainty** | `reports/figures/generated_by_ai/G6_forecast_uncertainty.png` | Predictive |
# | 24 | Recovery Simulator | `docs/streamlit/strategy/strategy_1.png` | Prescriptive |
# | 25 | Priority Matrix | `docs/streamlit/strategy/strategy_2.png` | Prescriptive |
# | 26 | Gantt Roadmap | `docs/streamlit/strategy/strategy_3.png` | Prescriptive |
#
# **Tong: 26 bieu do - 8 AI-generated - 4 cap do phan tich**
#
# ### Kha nang Tai lap
#
# - ✅ Random seed 42 (numpy + LightGBM + SHAP)
# - ✅ TRAIN_CUTOFF = 2022-12-31
# - ✅ Tat ca lag features dung shift(>=1)
# - ✅ External features dung lag >=365
# - ✅ TimeSeriesSplit CV voi gap 14 ngay
# - ✅ Iterative prediction voi FULL feature recomputation
# - ✅ Tat ca bieu do AI-generated co script tai `notebooks/06_output/generate_advanced_charts.py`
#
# ---
# **The GridBreakers - VinUni DS&AI Club - Datathon 2026**
#
# *Bao cao tong hop 26 truc quan hoa tren toan bo du an Gridbreaker, duoc to chuc theo Rubric 4 Cap do. 8 bieu do moi duoc AI tao ra de lap day khoang trong phan tich. Moi insight di kem Observation -> Why It Matters -> Strategy Link.*
