from sqlalchemy import create_engine

import pymysql

import pandas as pd



sqlEngine       = create_engine('mysql+pymysql://username:password@hostip/db', pool_recycle=3600)

dbConnection    = sqlEngine.connect()

frame           = pd.read_sql("select * from option_chain_datas limit 100 ", dbConnection);



pd.set_option('display.expand_frame_repr', False)

print(frame)

 

dbConnection.close()