import pandas as pd
import streamlit as st
import datetime
import pickle
# monthdic = {
#     'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
# }
# st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("MARKETING SPEND PREDICTION MODEL")
# st.subheader('Weekly')
st.write("")
st.write("Please enter values with respect to monthly data ...")

st.header('Feature Set 1')

orders2 = st.number_input(
    label="Monthly orders 1", step=1., format="%.4f")
st.write('You entered orders:', orders2)

revenue2 = st.number_input(
    label="Monthly revenue 1", step=1., format="%.4f")
st.write('You entered revenue:', revenue2)

averagePrice2 = st.number_input(
    label="Monthly average price 1", step=1., format="%.4f")
st.write('You entered Average Price:', averagePrice2)

facebookPurchases2 = st.number_input(
    label="Monthly facebook purchases 1", step=1., format="%.4f")
st.write('You entered facebook purchases:', facebookPurchases2)

facebookRevenue2 = st.number_input(
    label="Monthly facebook revenue 1", step=1., format="%.4f")
st.write('You entered Facebook Revenue:', facebookRevenue2)

googleSpend2 = st.number_input(
    label="Monthly google spend 1", step=1., format="%.4f")
st.write('You entered google spend:', googleSpend2)

googlePurchases2 = st.number_input(
    label="Monthly google purchases 1", step=1., format="%.4f")
st.write('You entered google purchases:', googlePurchases2)

googleRevenue2 = st.number_input(
    label="Monthly google revenue 1", step=1., format="%.4f")
st.write('You entered google revenue:', googleRevenue2)

facebookROAS2 = st.number_input(
    label="Monthly facebook ROAS 1", step=1., format="%.4f")
st.write('You entered facebook ROAS:', facebookROAS2)

googleCPA2 = st.number_input(
    label="Monthly google CPA 1", step=1., format="%.4f")
st.write('You entered google CPA:', googleCPA2)

googleROAS2 = st.number_input(
    label="Monthly google ROAS 1", step=1., format="%.4f")
st.write('You entered google ROAS:', googleROAS2)

totalROAS2 = st.number_input(
    label="Monthly total ROAS 1", step=1., format="%.4f")
st.write('You entered total ROAS:', totalROAS2)

targetRevenue2 = st.number_input(
    label="Monthly target revenue 1", step=1., format="%.4f")
st.write('You entered target revenue:', targetRevenue2)

#
# month = monthdic[Month]
# year = 2021
# Week_Number = #int(datetime.date(year,month , 1).strftime("%V"))
# M_ActualRevenue = targetRevenue  # (targetRevenue/30)*7


if orders2 > 0 and revenue2 > 0 and averagePrice2 > 0 and facebookPurchases2 > 0 and facebookRevenue2 > 0 and googleSpend2 > 0 and googlePurchases2 > 0 and googleRevenue2 > 0 and facebookROAS2 > 0 and googleCPA2 > 0 and googleROAS2 > 0 and totalROAS2 > 0 and targetRevenue2 > 0:
    st.write('----------------------------------- Marketing Spend Budget Prediction ----------------------------------------')

    # ['orders', 'revenue', 'average_price', 'fb_purchases', 'fb_revenue', 'ga_spend', 'ga_purchases', 'ga_revenue', 'fb_roas','ga_roas', 'ga_cpa', 'roas', 'target_revenue']

    test2 = {'orders': orders2, 'revenue': revenue2,
             'average_price': averagePrice2, 'fb_purchases': facebookPurchases2, 'fb_revenue': facebookRevenue2,
             'ga_spend': googleSpend2, 'ga_purchases': googlePurchases2, 'ga_revenue': googleRevenue2,
             'fb_roas': facebookROAS2, 'ga_cpa': googleCPA2, 'ga_roas': googleROAS2, 'roas': totalROAS2, 'target_revenue': targetRevenue2}

    test_DF2 = pd.DataFrame([test2])

    # path2 = '/./model_scaler/'
    inner2 = pickle.load(open('./model_scaler/inner_1.gz', 'rb'))
    outer2 = pickle.load(open('./model_scaler/outer.gz', 'rb'))

    model2 = pickle.load(open('./model_scaler/xgb_reg_1.pkl', "rb"))

    test_DF2 = pd.DataFrame(inner2.transform(test_DF2))
    test_DF2 = test_DF2.rename(columns={0: 'orders', 1: 'revenue', 2: 'average_price', 3: 'fb_purchases',
                                        4: 'fb_revenue', 5: 'ga_spend', 6: 'ga_purchases', 7: 'ga_revenue', 8: 'fb_roas', 9: 'ga_cpa', 10: 'ga_roas', 11: 'roas', 12: 'target_revenue'})

    pred2 = model2.predict(test_DF2)
    pred2 = pred2.reshape(-1, 1)
    pred2 = outer2.inverse_transform(pred2)
    totalFB_Revenue2 = float(pred2[0])
    totalGA_Revenue2 = float(totalFB_Revenue2*0.1)

    st.write('Facebook Marketing Spend Budget :', totalFB_Revenue2)
    st.write('Google Marketing Spend Budget :', totalGA_Revenue2)


#  ??????????????????????????????????????????????????????????????????????????????????????


# monthdic = {
#     'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
# }
# st.set_option('deprecation.showfileUploaderEncoding', False)

st.header('Feature Set 2')

orders3 = st.number_input(
    label="Monthly orders 2", step=1., format="%.4f")
st.write('You entered orders:', orders3)

revenue3 = st.number_input(
    label="Monthly revenue 2", step=1., format="%.4f")
st.write('You entered revenue:', revenue3)

averagePrice3 = st.number_input(
    label="Monthly average price 2", step=1., format="%.4f")
st.write('You entered Average Price:', averagePrice3)

facebookPurchases3 = st.number_input(
    label="Monthly facebook purchases 2", step=1., format="%.4f")
st.write('You entered facebook purchases:', facebookPurchases3)

facebookRevenue3 = st.number_input(
    label="Monthly facebook revenue 2", step=1., format="%.4f")
st.write('You entered Facebook Revenue:', facebookRevenue3)

googleSpend3 = st.number_input(
    label="Monthly google spend 2", step=1., format="%.4f")
st.write('You entered google spend:', googleSpend3)

googlePurchases3 = st.number_input(
    label="Monthly google purchases 2", step=1., format="%.4f")
st.write('You entered google purchases:', googlePurchases3)

googleRevenue3 = st.number_input(
    label="Monthly google revenue 2", step=1., format="%.4f")
st.write('You entered google revenue:', googleRevenue3)

facebookROAS3 = st.number_input(
    label="Monthly facebook ROAS 2", step=1., format="%.4f")
st.write('You entered facebook ROAS:', facebookROAS3)

googleCPA3 = st.number_input(
    label="Monthly google CPA 2", step=1., format="%.4f")
st.write('You entered google CPA:', googleCPA3)

googleROAS3 = st.number_input(
    label="Monthly google ROAS 2", step=1., format="%.4f")
st.write('You entered google ROAS:', googleROAS3)

totalROAS3 = st.number_input(
    label="Monthly total ROAS 2", step=1., format="%.4f")
st.write('You entered total ROAS:', totalROAS3)

#
# month = monthdic[Month]
# year = 2021
# Week_Number = #int(datetime.date(year,month , 1).strftime("%V"))
# M_ActualRevenue = targetRevenue  # (targetRevenue/30)*7


if orders3 > 0 and revenue3 > 0 and averagePrice3 > 0 and facebookPurchases3 > 0 and facebookRevenue3 > 0 and googleSpend3 > 0 and googlePurchases3 > 0 and googleRevenue3 > 0 and facebookROAS3 > 0 and googleCPA3 > 0 and googleROAS3 > 0 and totalROAS3 > 0:
    st.write('----------------------------------- Marketing Spend Budget Prediction ----------------------------------------')

    # ['orders', 'revenue', 'average_price', 'fb_purchases', 'fb_revenue', 'ga_spend', 'ga_purchases', 'ga_revenue', 'fb_roas', 'ga_cpa', 'ga_roas', 'roas', 'target_revenue']

    test3 = {'orders': orders3, 'revenue': revenue3,
             'average_price': averagePrice3, 'fb_purchases': facebookPurchases3, 'fb_revenue': facebookRevenue3,
             'ga_spend': googleSpend3, 'ga_purchases': googlePurchases3, 'ga_revenue': googleRevenue3,
             'fb_roas': facebookROAS3, 'ga_cpa': googleCPA3, 'ga_roas': googleROAS3, 'roas': totalROAS3}

    test_DF3 = pd.DataFrame([test3])

    # path3 = '/./model_scaler/'
    inner3 = pickle.load(open('./model_scaler/inner_2.gz', 'rb'))
    outer3 = pickle.load(open('./model_scaler/outer.gz', 'rb'))

    model3 = pickle.load(open('./model_scaler/xgb_reg_2.pkl', "rb"))

    test_DF3 = pd.DataFrame(inner3.transform(test_DF3))

    test_DF3 = test_DF3.rename(columns={0: 'orders', 1: 'revenue', 2: 'average_price', 3: 'fb_purchases',
                                        4: 'fb_revenue', 5: 'ga_spend', 6: 'ga_purchases', 7: 'ga_revenue', 8: 'fb_roas', 9: 'ga_cpa', 10: 'ga_roas', 11: 'roas'})

    pred3 = model3.predict(test_DF3)
    pred3 = pred3.reshape(-1, 1)
    pred3 = outer3.inverse_transform(pred3)
    totalFB_Revenue3 = float(pred3[0])
    totalGA_Revenue3 = float(totalFB_Revenue3*0.1)

    st.write('Facebook Marketing Spend Budget :', totalFB_Revenue3)
    st.write('Google Marketing Spend Budget :', totalGA_Revenue3)


# ?????????????????????????????????????????????????????????????????????


# monthdic = {
#     'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
# }
# st.set_option('deprecation.showfileUploaderEncoding', False)

st.header('Feature Set 3')

orders4 = st.number_input(
    label="Monthly orders 3", step=1., format="%.4f")
st.write('You entered orders:', orders4)

revenue4 = st.number_input(
    label="Monthly revenue 3", step=1., format="%.4f")
st.write('You entered revenue:', revenue4)

averagePrice4 = st.number_input(
    label="Monthly average price 3", step=1., format="%.4f")
st.write('You entered Average Price:', averagePrice4)

facebookPurchases4 = st.number_input(
    label="Monthly facebook purchases 3", step=1., format="%.4f")
st.write('You entered facebook purchases:', facebookPurchases4)

facebookRevenue4 = st.number_input(
    label="Monthly facebook revenue 3", step=1., format="%.4f")
st.write('You entered Facebook Revenue:', facebookRevenue4)

googleSpend4 = st.number_input(
    label="Monthly google spend 3", step=1., format="%.4f")
st.write('You entered google spend:', googleSpend4)

googlePurchases4 = st.number_input(
    label="Monthly google purchases 3", step=1., format="%.4f")
st.write('You entered google purchases:', googlePurchases4)

googleRevenue4 = st.number_input(
    label="Monthly google revenue 3", step=1., format="%.4f")
st.write('You entered google revenue:', googleRevenue4)

facebookROAS4 = st.number_input(
    label="Monthly facebook ROAS 3", step=1., format="%.4f")
st.write('You entered facebook ROAS:', facebookROAS4)

googleROAS4 = st.number_input(
    label="Monthly google ROAS 3", step=1., format="%.4f")
st.write('You entered google ROAS:', googleROAS4)

totalROAS4 = st.number_input(
    label="Monthly total ROAS 3", step=1., format="%.4f")
st.write('You entered total ROAS:', totalROAS4)

#
# month = monthdic[Month]
# year = 2021
# Week_Number = #int(datetime.date(year,month , 1).strftime("%V"))
# M_ActualRevenue = targetRevenue  # (targetRevenue/30)*7


if orders4 > 0 and revenue4 > 0 and averagePrice4 > 0 and facebookPurchases4 > 0 and facebookRevenue4 > 0 and googleSpend4 > 0 and googlePurchases4 > 0 and googleRevenue4 > 0 and facebookROAS4 > 0 and googleROAS4 > 0 and totalROAS4 > 0:
    st.write('----------------------------------- Marketing Spend Budget Prediction ----------------------------------------')

    # ['orders', 'revenue', 'average_price', 'fb_purchases', 'fb_revenue', 'ga_spend', 'ga_purchases', 'ga_revenue', 'fb_roas', 'ga_cpa', 'ga_roas', 'roas', 'target_revenue']

    test4 = {'orders': orders4, 'revenue': revenue4,
             'average_price': averagePrice4, 'fb_purchases': facebookPurchases4, 'fb_revenue': facebookRevenue4,
             'ga_spend': googleSpend4, 'ga_purchases': googlePurchases4, 'ga_revenue': googleRevenue4,
             'fb_roas': facebookROAS4, 'ga_roas': googleROAS4, 'roas': totalROAS4}

    test_DF4 = pd.DataFrame([test4])

    # path4 = '/./model_scaler/'
    inner4 = pickle.load(open('./model_scaler/inner_3.gz', 'rb'))
    outer4 = pickle.load(open('./model_scaler/outer.gz', 'rb'))

    model4 = pickle.load(open('./model_scaler/xgb_reg_3.pkl', "rb"))

    test_DF4 = pd.DataFrame(inner4.transform(test_DF4))

    test_DF4 = test_DF4.rename(columns={0: 'orders', 1: 'revenue', 2: 'average_price', 3: 'fb_purchases',
                                        4: 'fb_revenue', 5: 'ga_spend', 6: 'ga_purchases', 7: 'ga_revenue', 8: 'fb_roas', 9: 'ga_roas', 10: 'roas'})

    pred4 = model4.predict(test_DF4)
    pred4 = pred4.reshape(-1, 1)
    pred4 = outer4.inverse_transform(pred4)
    totalFB_Revenue4 = float(pred4[0])
    totalGA_Revenue4 = float(totalFB_Revenue4*0.1)

    st.write('Facebook Marketing Spend Budget :', totalFB_Revenue4)
    st.write('Google Marketing Spend Budget :', totalGA_Revenue4)


#  ????????????????????????????????????????????????????????????????????????????


# monthdic = {
#     'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
# }
# st.set_option('deprecation.showfileUploaderEncoding', False)

st.header('Feature Set 4')

revenue5 = st.number_input(
    label="Monthly revenue 4", step=1., format="%.4f")
st.write('You entered revenue:', revenue5)

averagePrice5 = st.number_input(
    label="Monthly average price 4", step=1., format="%.4f")
st.write('You entered Average Price:', averagePrice5)

facebookPurchases5 = st.number_input(
    label="Monthly facebook purchases 4", step=1., format="%.4f")
st.write('You entered facebook purchases:', facebookPurchases5)

facebookRevenue5 = st.number_input(
    label="Monthly facebook revenue 4", step=1., format="%.4f")
st.write('You entered Facebook Revenue:', facebookRevenue5)

googleSpend5 = st.number_input(
    label="Monthly google spend 4", step=1., format="%.4f")
st.write('You entered google spend:', googleSpend5)

googlePurchases5 = st.number_input(
    label="Monthly google purchases 4", step=1., format="%.4f")
st.write('You entered google purchases:', googlePurchases5)

googleRevenue5 = st.number_input(
    label="Monthly google revenue 4", step=1., format="%.4f")
st.write('You entered google revenue:', googleRevenue5)

facebookROAS5 = st.number_input(
    label="Monthly facebook ROAS 4", step=1., format="%.4f")
st.write('You entered facebook ROAS:', facebookROAS5)

googleROAS5 = st.number_input(
    label="Monthly google ROAS 4", step=1., format="%.4f")
st.write('You entered google ROAS:', googleROAS5)

totalROAS5 = st.number_input(
    label="Monthly total ROAS 4", step=1., format="%.4f")
st.write('You entered total ROAS:', totalROAS5)

#
# month = monthdic[Month]
# year = 2021
# Week_Number = #int(datetime.date(year,month , 1).strftime("%V"))
# M_ActualRevenue = targetRevenue  # (targetRevenue/30)*7


if revenue5 > 0 and averagePrice5 > 0 and facebookPurchases5 > 0 and facebookRevenue5 > 0 and googleSpend5 > 0 and googlePurchases5 > 0 and googleRevenue5 > 0 and facebookROAS5 > 0 and googleROAS5 > 0 and totalROAS5 > 0:
    st.write('----------------------------------- Marketing Spend Budget Prediction ----------------------------------------')

    # ['orders', 'revenue', 'average_price', 'fb_purchases', 'fb_revenue', 'ga_spend', 'ga_purchases', 'ga_revenue', 'fb_roas', 'ga_cpa', 'ga_roas', 'roas', 'target_revenue']

    test5 = {'revenue': revenue5,
             'average_price': averagePrice5, 'fb_purchases': facebookPurchases5, 'fb_revenue': facebookRevenue5,
             'ga_spend': googleSpend5, 'ga_purchases': googlePurchases5, 'ga_revenue': googleRevenue5,
             'fb_roas': facebookROAS5, 'ga_roas': googleROAS5, 'roas': totalROAS5}

    test_DF55 = pd.DataFrame([test5])

    # path5 = '/./model_scaler/'
    inner5 = pickle.load(open('./model_scaler/inner_4.gz', 'rb'))
    outer5 = pickle.load(open('./model_scaler/outer.gz', 'rb'))

    model5 = pickle.load(open('./model_scaler/xgb_reg_4.pkl', "rb"))

    test_DF5 = pd.DataFrame(inner5.transform(test_DF55))

    test_DF5 = test_DF5.rename(columns={0: 'revenue', 1: 'average_price', 2: 'fb_purchases',
                                        3: 'fb_revenue', 4: 'ga_spend', 5: 'ga_purchases', 6: 'ga_revenue', 7: 'fb_roas', 8: 'ga_roas', 9: 'roas'})

    pred5 = model5.predict(test_DF5)
    pred5 = pred5.reshape(-1, 1)
    pred5 = outer5.inverse_transform(pred5)
    totalFB_Revenue5 = float(pred5[0])
    totalGA_Revenue5 = float(totalFB_Revenue5*0.1)

    st.write('Facebook Marketing Spend Budget :', totalFB_Revenue5)
    st.write('Google Marketing Spend Budget :', totalGA_Revenue5)

# ??????????????????????????????????????????????????????????????????????


# monthdic = {
#     'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
# }
# st.set_option('deprecation.showfileUploaderEncoding', False)

st.header('Feature Set 5')

revenue6 = st.number_input(
    label="Monthly revenue 5", step=1., format="%.4f")
st.write('You entered revenue:', revenue6)

averagePrice6 = st.number_input(
    label="Monthly average price 5", step=1., format="%.4f")
st.write('You entered Average Price:', averagePrice6)

facebookPurchases6 = st.number_input(
    label="Monthly facebook purchases 5", step=1., format="%.4f")
st.write('You entered facebook purchases:', facebookPurchases6)

facebookRevenue6 = st.number_input(
    label="Monthly facebook revenue 5", step=1., format="%.4f")
st.write('You entered Facebook Revenue:', facebookRevenue6)

googleSpend6 = st.number_input(
    label="Monthly google spend 5", step=1., format="%.4f")
st.write('You entered google spend:', googleSpend6)

googlePurchases6 = st.number_input(
    label="Monthly google purchases 5", step=1., format="%.4f")
st.write('You entered google purchases:', googlePurchases6)

facebookROAS6 = st.number_input(
    label="Monthly facebook ROAS 5", step=1., format="%.4f")
st.write('You entered facebook ROAS:', facebookROAS6)

googleROAS6 = st.number_input(
    label="Monthly google ROAS 5", step=1., format="%.4f")
st.write('You entered google ROAS:', googleROAS6)

totalROAS6 = st.number_input(
    label="Monthly total ROAS 5", step=1., format="%.4f")
st.write('You entered total ROAS:', totalROAS6)

#
# month = monthdic[Month]
# year = 2021
# Week_Number = #int(datetime.date(year,month , 1).strftime("%V"))
# M_ActualRevenue = targetRevenue  # (targetRevenue/30)*7


if revenue6 > 0 and averagePrice6 > 0 and facebookPurchases6 > 0 and facebookRevenue6 > 0 and googleSpend6 > 0 and googlePurchases6 > 0 and facebookROAS6 > 0 and googleROAS6 > 0 and totalROAS6 > 0:
    st.write('----------------------------------- Marketing Spend Budget Prediction ----------------------------------------')

    # ['orders', 'revenue', 'average_price', 'fb_purchases', 'fb_revenue', 'ga_spend', 'ga_purchases', 'ga_revenue', 'fb_roas', 'ga_cpa', 'ga_roas', 'roas', 'target_revenue']

    test6 = {'revenue': revenue6,
             'average_price': averagePrice6, 'fb_purchases': facebookPurchases6, 'fb_revenue': facebookRevenue6,
             'ga_spend': googleSpend6, 'ga_purchases': googlePurchases6,
             'fb_roas': facebookROAS6, 'ga_roas': googleROAS6, 'roas': totalROAS6}

    test_DF6 = pd.DataFrame([test6])

    # path6 = '/./model_scaler/'
    inner6 = pickle.load(open('./model_scaler/inner_5.gz', 'rb'))
    outer6 = pickle.load(open('./model_scaler/outer.gz', 'rb'))

    model6 = pickle.load(open('./model_scaler/xgb_reg_5.pkl', "rb"))

    test_DF6 = pd.DataFrame(inner6.transform(test_DF6))

    test_DF6 = test_DF6.rename(columns={0: 'revenue', 1: 'average_price', 2: 'fb_purchases',
                                        3: 'fb_revenue', 4: 'ga_spend', 5: 'ga_purchases', 6: 'fb_roas', 7: 'ga_roas', 8: 'roas'})

    pred6 = model6.predict(test_DF6)
    pred6 = pred6.reshape(-1, 1)
    pred6 = outer6.inverse_transform(pred6)
    totalFB_Revenue6 = float(pred6[0])
    totalGA_Revenue6 = float(totalFB_Revenue6*0.1)

    st.write('Facebook Marketing Spend Budget :', totalFB_Revenue6)
    st.write('Google Marketing Spend Budget :', totalGA_Revenue6)


['orders', 'revenue', 'average_price', 'fb_purchases', 'fb_revenue', 'ga_spend',
    'ga_purchases', 'ga_revenue', 'fb_roas', 'ga_cpa', 'ga_roas', 'roas', 'target_revenue']
['orders', 'revenue', 'average_price', 'fb_purchases', 'fb_revenue', 'ga_spend',
    'ga_purchases', 'ga_revenue', 'fb_roas', 'ga_roas', 'ga_cpa', 'roas', 'target_revenue']
