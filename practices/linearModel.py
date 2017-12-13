import tensorflow as tf

#modal params PS:wont init until tf.global_variables_initializer call
W = tf.Variable([.3],dtype=tf.float32)
b = tf.Variable([-.3],dtype=tf.float32)

#input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

#loss
loss = tf.reduce_sum(tf.square(linear_model - y))
#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01) #change this value to see what's difference
train = optimizer.minimize(loss)

#training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

init = tf.global_variables_initializer()  #init W,B

session = tf.Session()
session.run(init)

for i in range(1000):
    session.run(train,{x:x_train,y:y_train})

curr_W, curr_b, curr_loss = session.run([W,b,loss],{x:x_train,y:y_train})

print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
