# -*- coding: utf-8 -*-
"""
Spyder Editor
此练习主要配合“TensorFlow的基本使用.ppt”
This is a temporary script file.
"""


#%%  概念1：Tensor

import tensorflow as tf 
a = tf.constant([1.0,2.0,3.0],name="Variable_a") 
b = tf.constant([1.0,2.0,3.0],name="Variable_b") 
c = tf.constant([[1,2,3],[4,5,6]],name="Variable_c") 
result = tf.add(a,b,name="Variable_result")
print(a)
print(b)
print(c)
print(result)


#%%  概念2：计算图

import tensorflow as tf 
my_log_path='C:\\tem'  
a = tf.constant([1.0,2.0,3.0],name="Variable_a") 
b = tf.constant([1.0,2.0,3.0],name="Variable_b") 
result = tf.add(a,b,name="my_add")
writer = tf.summary.FileWriter(my_log_path, tf.get_default_graph())
writer.close()
print(tf.get_default_graph().as_graph_def())
print(result)
##tensorboard --logdir=C://tem


#%%  概念3：构建图 概念4：在一个会话中启动图

#import tensorflow as tf 
#my_log_path='C:\\tem'  
#a = tf.constant([1.0,2.0,3.0],name="Variable_a") 
#b = tf.constant([1.0,2.0,3.0],name="Variable_b") 
#result = tf.add(a,b,name="my_add")
#sess = tf.Session() 
#value_a = sess.run(a)
#value_b = sess.run(b)
#value_result = sess.run(result)
#print(a)   
##print(sess.run(a))
#sess.close()


import tensorflow as tf 
my_log_path='C:\\tem'  
a = tf.constant([1.0,2.0,3.0],name="Variable_a") 
b = tf.constant([1.0,2.0,3.0],name="Variable_b") 
result = tf.add(a,b,name="my_add")
with tf.Session() as sess:
    value_a = sess.run(a)
    value_b = sess.run(b)
    value_result = sess.run(result)
    print(a)   
    print(sess.run(a))


#%% 概念5：变量

import tensorflow as tf 
W = tf.Variable(10)
assign_op = W.assign(100)

with tf.Session() as sess:
#	sess.run(W.initializer)
	sess.run(tf.global_variables_initializer())
	a=sess.run(W)    
	print(W.eval()) 
	print(sess.run(assign_op)) 

#%%

#%% 变量在不同的graph中的实现
import tensorflow as tf 
W = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()

# You have to initialize W at each session
sess1.run(W.initializer)
sess2.run(W.initializer)

print(sess1.run(W.assign_add(10))) # >> 20
print(sess2.run(W.assign_sub(2))) # >> 8

print(sess1.run(W.assign_add(100))) # >> 120
print(sess2.run(W.assign_sub(50))) # >> -42

sess1.close()
sess2.close()

#%% 概念6：Fetch
import tensorflow as tf 
a = tf.constant([2.0],name="Variable_a") 
b = tf.constant([3.0],name="Variable_b") 
c = tf.constant([4.0],name="Variable_b") 
result_1 = tf.add(a,b)
result_2 = tf.multiply(result_1,c)
with tf.Session() as sess:
    result = sess.run([result_1,result_2])


#%% 概念7：Feed
import tensorflow as tf
a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant([5, 5, 5], tf.float32)
c = a + b
with tf.Session() as sess:
    print(sess.run(c, {a: [1, 2, 3]})) 
    


    