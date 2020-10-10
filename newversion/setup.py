import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import bs4 as bs
import requests
import yfinance as yf
import datetime
import io
import cv2
import skimage
import datetime
from PIL import Image
from pandas_datareader import data as pdr
from skimage import measure
from skimage.measure import block_reduce
from datetime import datetime

'''
Functions to be used for data generation 
'''

def get_img_from_fig(fig, dpi=180):
   # get_img_from_fig is function which returns an image as numpy array from figure
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def build_image(stockindex, idate=10, pastlag=10, futlag=3,nb_dates=1000):
  # Build image from a table stockindex price list 
  #return a (32,32,3) np.array representing the image in color
  #ising idate as a starting point
  #paslag futlag number of days to consider for translate
  sp500close=stockindex
  nb_days=nb_dates

  x_datas=[]
  x_datas=np.zeros((32,32,3))
  i=idate
  
  fig=plt.figure()
  ax=fig.add_subplot(111)
  ax.plot(sp500close[(i-pastlag):i])
  plot_img_np = get_img_from_fig(fig)
  x_tmp= skimage.measure.block_reduce(plot_img_np[90:620,140:970], (18,28,1), np.mean)
  (x_datas[1:-1])[:,1:-1][:]=x_tmp
  fig.clear()
  plt.close(fig)
    
  x_datas=x_datas/255
  return x_datas

  
'''
MAIN FUNCTION OF CLASSIFICATION 
build y state y fut 
and x  
'''
def class_shortterm_returnfut(x, yfut, indexforpast,tpastlag):
  '''
  #this function is use to classifiy the future state based on the position of future value with the past range 
  #Put the value from the 2 boxes (max min) or (open close) on the time range  and check if it is within
  #go down go up or exit the box
  #the fucntion return 5 state depending on the future value position on the boxes and one state for error cases
  '''

  xpast_min=np.min(x[(indexforpast-tpastlag):indexforpast])
  xpast_max=np.max(x[(indexforpast-tpastlag):indexforpast])
  x_open=x[int(indexforpast-tpastlag)]
  x_close=x[indexforpast]
  
  if (yfut < xpast_min ): return 0
  elif  (yfut < min(x_open,x_close)): return 1
  elif  (yfut < max(x_open,x_close)): return 2
  elif  (yfut < xpast_max): return 3
  elif  (yfut > xpast_max): return 4
  else  : return -1

def main_class_shortterm_returnfut(iterable):
  return class_shortterm_returnfut(sp500close, iterable, pastlag,futlag)

def normalise_df_image(xdf):
  #normalisation to 0,1 range of the equity index
  df_tmp=xdf
  maxval=np.max(df_tmp)
  df_tmp=df_tmp/maxval
  return df_tmp, maxval

def build_image(stockindex, idate=10, pastlag=10, futlag=3):
  #another version of returning image from a data frame index
  #using the pastlag as range for the graph
  #ising idate as a starting point
  #return a (32,32,3) np array

  #number of days to consider for translate
  sp500close=stockindex
  x_datas=[]
  x_datas=np.zeros((32,32,3))
  i=idate
  
  fig=plt.figure()
  ax=fig.add_subplot(111)
  ax.plot(sp500close[(i-pastlag):i])
  plot_img_np = get_img_from_fig(fig)
  x_tmp= skimage.measure.block_reduce(plot_img_np[90:620,140:970], (18,28,1), np.mean)
  (x_datas[1:-1])[:,1:-1][:]=x_tmp
  fig.clear()
  plt.close(fig)
    
  x_datas=x_datas/255
  return x_datas

def build_image_df(xdf, past_step,fut_step) :
  '''
  returning a dictionary of time series dataframes to be used in setup_input_NN_image so a to generate 
  Input X Result Y_StateClass, Y_FutPredict
  pastlag as range for the graph
  fut _step the future value lag in time to predict or to check the financial state of the market 
  #times series to get information from the stock index value
  'stock_value':the time serie of the index normalised on the whole period
  'moving_average':  time serie of the rolling moving average value of the index for past step image
  "max": time serie of the rolling max  value of the index for past step image
  "min": time serie of the rolling  min value of the index for past step image
  'volatility':  time serie of the rolling  vol value of the index for past step image
          
  'df_x_image': is a time series of flattened (1, ) calculed from images (32, 32, 3) list 
  #I had to flatten it because panda does not create table with this format
  'market_state': future markket state to be predicted time lag is futlag
  'future_value': future value of stock price to predict  time lag is futlag
  'future_volatility':  time serie of the future volatility of the index time lag is futlag
  '''

  df_stockvaluecorrected=xdf
  df_stockvaluecorrected, _ = normalise_df_image(df_stockvaluecorrected)
  df_pctchge = df_stockvaluecorrected.pct_change(periods=past_step)
  df_movave = df_stockvaluecorrected.rolling(window=past_step).mean()
  df_volaty = np.sqrt(252)*df_pctchge.rolling(window=past_step).std()
  df_max =df_stockvaluecorrected.rolling(window=past_step).max()
  df_min =df_stockvaluecorrected.rolling(window=past_step).min()
  df_Fut_value =df_stockvaluecorrected.shift(periods=-fut_step)
  df_Fut_value.name='future_value'
  df_Fut_volaty =df_volaty.shift(periods=-fut_step)
  
  df_market_state=pd.DataFrame(index=df_stockvaluecorrected.index,columns=['market_state'],dtype=np.float64)
  
  tmpimage=np.zeros((255,255))
  flatten_image=np.reshape(tmpimage,(1,-1))
  colname_d_x_image_flattened = ['Image Col'+str(j) for j in range(flatten_image.shape[1])]

  np_x_image=np.zeros((len(df_stockvaluecorrected.index),flatten_image.shape[1]))
  
  for i in range(len(df_stockvaluecorrected.index)):
        yfut=df_Fut_value.iloc[i]
        df_market_state.iloc[i]=class_shortterm_returnfut(df_stockvaluecorrected,yfut, i,tpastlag=past_step)
        print("loop 1 market state :", "step ",i,"market state fut", df_market_state.iloc[i]," future value",df_Fut_value.iloc[i] )
  df_market_state.index=df_Fut_value.index

  fig=plt.figure()
  
  '''
  for i in range(len(df_stockvaluecorrected.index)):
        try:
          tmpimage=build_image_optimfig(fig, df_stockvaluecorrected,i,pastlag=past_step,futlag=fut_step)
          np_x_image[i,:]=np.reshape(tmpimage,(1,-1))
          print("loop 2 image :", "step ",i,"market state fut", df_market_state.iloc[i]," future value",df_Fut_value.iloc[i] )
        except:
           print("error at index", i)
  '''           
  
  def build_image_optimfig_simplified(i_index):
    return build_image_optimfig(fig, df_stockvaluecorrected,i_index,pastlag=past_step,futlag=fut_step)

  def quick_build_image_from_index(indexstart, index_end, np_x_image):
        if (indexstart==index_end):
            tmpimage=build_image_optimfig_simplified(indexstart)
            np_x_image[indexstart,:]=np.reshape(tmpimage,(1,-1))
            print("loop 2 image :", "step ",indexstart)
        else :
            i_split=indexstart+(index_end-indexstart)//2
            quick_build_image_from_index(indexstart, i_split,np_x_image)
            quick_build_image_from_index(i_split+1, index_end,np_x_image)

  quick_build_image_from_index(0, len(df_stockvaluecorrected.index)-1, np_x_image)

  df_x_image=pd.DataFrame(data=np_x_image,columns=colname_d_x_image_flattened, index=df_stockvaluecorrected.index)
  fig.clear
  plt.close(fig)


  df_data= {
          'stock_value': df_stockvaluecorrected, 
          'moving_average': df_movave, 
          "max": df_max, 
          "min": df_max,
          'volatility': df_volaty,
          'future_volatility': df_Fut_volaty,
          
          'df_x_image':df_x_image,
          'market_state':df_market_state,
          'future_value': df_Fut_value,

          }

  return df_data

def build_image_clean(stockindex_ohlcv, ret_image_size=(32,32,3), idate=10, pastlag=32):
  '''
  TO BE COMPLETED
  NOT USED NOW
  
  change one date into an array (32,32,3)
  Each absciss pixel is one day
  in ordinate the min value of ohlc shall be 0 (volume is tabled on the third image) 
  in ordinate the max value of ohlc shall be  (volume is tabled on the third image) 
  1st image: 32 x32
    based on each day we place the open and close point
    in ordinate int (255 * price /max ohlc)
    with value of  255 for close and 127 for open
  2nd image: 32 x32
    based on each day we place the high low point 
    in ordinate int (255 * price /max ohlc)
    with 64 for high and 32 for low
  3rd image: 32 x32
    each column value is a equal to int 255* volume of day / volume max period)
  '''
  #number of days to consider for translate
  tsindexstock=stockindex_ohlcv.iloc[(idate-pastlag):idate]
  valmax=np.max(np.array(tsindexstock[tsindexstock.columns[:-1]]))
  valmin=np.min(np.array(tsindexstock[tsindexstock.columns[:-1]]))
  vol=tsindexstock[tsindexstock.columns[-1]]
  
  x_datas=np.zeros(ret_image_size)
  
  return x_datas
  
def setup_input_NN_image(xdf, past_step=25,fut_step=5, split=0.8):
  '''
  this function the time serie of the index price 
  and generate the random dataset with split value from the whole time serie
  X is a time serie of the flattened 32, 32 ,3 image list
  Y_StateClass is a time serie of future state to predict with a classification made with class_shortterm_returnfut
  Y_FutPredict is the time serie of stocke index shifted in time to be predicted
  we randomize the dates and retun 2 set of dataframes
  '''
  xdf_data=build_image_df(xdf,past_step,fut_step)
  
  tmp_data=pd.concat([xdf_data['market_state'],xdf_data['future_value'],xdf_data['df_x_image']],axis=1)
  tmp_data=tmp_data.dropna()

  Y_StateClass= tmp_data['market_state']
  Y_FutPredict= tmp_data['future_value']  
  X=tmp_data.drop(columns=['market_state','future_value'])

  nb_dates=len(Y_StateClass.index)
  rng = np.random.default_rng()
  list_shuffle = np.arange(nb_dates)
  rng.shuffle(list_shuffle)
  split_index=int(split*nb_dates)
    
  train_split=list_shuffle[:split_index]
  test_split=list_shuffle[(split_index+1):]

  X_train=(X.iloc[train_split])
  Y_train_StateClass=(Y_StateClass.iloc[train_split])
  Y_train_FutPredict=(Y_FutPredict.iloc[train_split])

  X_test=(X.iloc[test_split])
  Y_test_StateClass=(Y_StateClass.iloc[test_split])
  Y_test_FutPredict=(Y_FutPredict.iloc[test_split])

  return (X_train, Y_train_StateClass, Y_train_FutPredict), (X_test, Y_test_StateClass, Y_test_FutPredict)

def change_X_df__nparray_image(df_X_train_image_flattened ):
  '''
  setup_input_NN_image returns a dataframe of flaten image for x train and xtest
  then this function will change each date into a nparray list of images with 32, 32, 3 size 
  '''
  X_train_image=df_X_train_image_flattened
  nb_train=len(X_train_image.index)
  
  x_train=np.zeros((nb_train,255,255,1))
  for i in range(nb_train):
    tmp=np.array(X_train_image.iloc[i])
    tmp=tmp.reshape(255,255,1)
    x_train[i]=tmp
  return x_train

def build_image_optimfig(fig, stockindex, idate=10, pastlag=10, futlag=3):
  '''
  #version of returning image from a data frame index
  #using the pastlag as range for the graph
  #ising idate as a starting point
  #return a (32,32,3) np array
  #this one is optimisng the use of ram 
  '''

  #number of days to consider for translate
  sp500close=stockindex
  x_datas=[]
  x_datas=np.zeros((255,255,3))
  i=idate
  
  plt.plot(sp500close[(i-pastlag):i])
  plot_img_np = get_img_from_fig(fig)
  #x_tmp= skimage.measure.block_reduce(plot_img_np[90:620,140:970], (18,28,1), np.mean)
  x_tmp= skimage.measure.block_reduce(plot_img_np[90:620,140:970], (2,3,1), np.mean)
  (x_datas[:])[:,:][:]=(x_tmp[5:-5])[:,11:-11][:]
    
  x_datas=x_datas[:,:,0]/255
  return x_datas     


'''
COMMAND NOW FOR DOWNLOADING HISTORICAL DATAS FOR SP500
'''

#Recuperation from yahoo of sp500 large history
start = datetime(1920,1,1)
end = datetime(2020,7,31)
yf.pdr_override() # <== that's all it takes :-)
sp500 = pdr.get_data_yahoo('^GSPC', 
                           start,
                             end)

#generate the dataset it can take 6 - 8 hours
#Need to be optimzed with more time
testsp500=(sp500['Close'])[:]

_ , (X_test_image, Y_test_StateClass_image, Y_test_FutPredict_image) = setup_input_NN_image(testsp500, split=0)

#classification of the states to get a equally distributed dataset
y_StateClass_image=Y_test_StateClass_image
x_image=X_test_image
y_futurepredict_image=Y_test_FutPredict_image
#group by y state the x_image 
#count the min of the of each state 
#construct a directory for each block like cat, dog etc

non_monotonic_index =pd.Index(list(y_StateClass_image))

def localize_index_from_state(non_monotonic_index, state=0):
  state_loc=non_monotonic_index.get_loc(state)
  return [i for i in range(0,state_loc.size) if state_loc[i]]

try : 
  state_error_loc=localize_index_from_state(non_monotonic_index,-1) 
  y_StateClass_image_error =y_StateClass_image.iloc[state_error_loc]
  x_image_State_is_error =x_image.iloc[state_error_loc].head()
except :
  print("No value for error state")

state_zero_loc=localize_index_from_state(non_monotonic_index, 0)
state_one_loc=localize_index_from_state(non_monotonic_index, 1)
state_two_loc=localize_index_from_state(non_monotonic_index, 2)
state_three_loc=localize_index_from_state(non_monotonic_index, 3)
state_four_loc=localize_index_from_state(non_monotonic_index, 4)

#Build up class for the dataset
y_StateClass_image_0 =y_StateClass_image.iloc[state_zero_loc]
y_StateClass_image_1 =y_StateClass_image.iloc[state_one_loc]
y_StateClass_image_2 =y_StateClass_image.iloc[state_two_loc]
y_StateClass_image_3 =y_StateClass_image.iloc[state_three_loc]
y_StateClass_image_4 =y_StateClass_image.iloc[state_four_loc]

x_image_State_is_0 =x_image.iloc[state_zero_loc]
x_image_State_is_1 =x_image.iloc[state_one_loc]
x_image_State_is_2 =x_image.iloc[state_two_loc]
x_image_State_is_3 =x_image.iloc[state_three_loc]
x_image_State_is_4 =x_image.iloc[state_four_loc]

y_futpredict_image_0 =y_futurepredict_image.iloc[state_zero_loc]
y_futpredict_image_1 =y_futurepredict_image.iloc[state_one_loc]
y_futpredict_image_2 =y_futurepredict_image.iloc[state_two_loc]
y_futpredict_image_3 =y_futurepredict_image.iloc[state_three_loc]
y_futpredict_image_4 =y_futurepredict_image.iloc[state_four_loc]

#print size of each dataset
print("dataset class 0 size is :",y_StateClass_image_0.size, "and for x ", x_image_State_is_0.index.size)
print("dataset class 1 size is :",y_StateClass_image_1.size, "and for x ", x_image_State_is_1.index.size)
print("dataset class 2 size is :",y_StateClass_image_2.size, "and for x ", x_image_State_is_2.index.size)
print("dataset class 3 size is :",y_StateClass_image_3.size, "and for x ", x_image_State_is_3.index.size)
print("dataset class 4 size is :",y_StateClass_image_4.size, "and for x ", x_image_State_is_4.index.size)

#write dataset for each set  in corresponding folder
def print_data_class(state=0,write_path='datas/state_is_') :
  state_zero_loc=localize_index_from_state(non_monotonic_index, state)
  y_StateClass_image_0 =y_StateClass_image.iloc[state_zero_loc]
  x_image_State_is_0 =x_image.iloc[state_zero_loc]
  y_futpredict_image_0 =y_futurepredict_image.iloc[state_zero_loc]
  y_StateClass_image_0.to_csv(write_path+str(state)+'/y_stateclass.csv')
  x_image_State_is_0.to_csv(write_path+str(state)+'/x_image.csv')
  y_futpredict_image_0.to_csv(write_path+str(state)+'/y_future.csv')

print_data_class(state=0)
print_data_class(state=1)
print_data_class(state=2)
print_data_class(state=3)
print_data_class(state=4)

#we need dataset of train, validation, test 
