# Name: Rajarshi Haldar
# Roll No.: 14CS10038
import numpy as np
import tensorflow as tf
import argparse
from zipfile import ZipFile
import math
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from sklearn.linear_model import LogisticRegression
from os import path
from glob import glob

def load_data(mode = 'train'):
    label_filename = mode + '_labels'
    image_filename = mode + '_images'
    label_zip = '../data/' + label_filename + '.zip'
    image_zip = '../data/' + image_filename + '.zip'
    with ZipFile(label_zip, 'r') as lblzip:
        labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8)
    with ZipFile(image_zip, 'r') as imgzip:
        images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    return images, labels


class NN(object):
    def __init__(self,input_dim, output_dim, hidden_dim, learning_rate):
        self.graph = tf.Graph()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.update = True

        with self.graph.as_default() :
            self.X = tf.placeholder(tf.float32, [None, 784], name="X")
            self.Y = tf.placeholder(tf.float32, [None, 10], name="Y")

            self.w1 = tf.Variable(tf.truncated_normal([784, self.hidden_dim], stddev = 1/math.sqrt(self.input_dim) , seed = tf.set_random_seed(12345)), name="w1")
            self.b1 = tf.Variable(tf.zeros([self.hidden_dim]), name="b1")
            self.h1 = tf.matmul(self.X, self.w1) + self.b1
            self.h1 = self.my_relu(self.h1)

            self.w2 = tf.Variable(tf.truncated_normal([self.hidden_dim, self.hidden_dim], stddev = 1/math.sqrt(self.hidden_dim) , seed = tf.set_random_seed(12345)), name="w2")
            self.b2 = tf.Variable(tf.zeros([self.hidden_dim]), name="b2")
            self.h2 = tf.matmul(self.h1, self.w2) + self.b2
            self.h2 = self.my_relu(self.h2)

            self.w3 = tf.Variable(tf.truncated_normal([self.hidden_dim, self.hidden_dim], stddev = 1/math.sqrt(self.hidden_dim) , seed = tf.set_random_seed(12345)), name="w3")
            self.b3 = tf.Variable(tf.zeros([self.hidden_dim]), name="b3")
            self.h3 = tf.matmul(self.h2, self.w3) + self.b3
            self.h3 = self.my_relu(self.h3)

            self.w4 = tf.Variable(tf.truncated_normal([self.hidden_dim, self.output_dim], stddev = 1/math.sqrt(self.hidden_dim) , seed = tf.set_random_seed(12345)), name="w4")
            self.b4 = tf.Variable(tf.zeros([self.output_dim]), name="b4")
            self.output = tf.matmul(self.h3, self.w4) + self.b4
            self.pred = self.my_softmax(self.output) #tf.nn.softmax(self.output)
            # self.pred = tf.nn.softmax(self.output)
            self.correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name="Accuracy")
            tf.summary.scalar('accuracy', self.accuracy)
            self.cost = self.cross_entropy_loss(self.pred, self.Y)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            tf.summary.scalar('mean_loss', self.cost)
            self.merged = tf.summary.merge_all()

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):                     
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost, global_step=self.global_step)

            
    def my_relu(self, layer, update = True):
        alpha = ops.convert_to_tensor(0, dtype=layer.dtype, name="alpha")
        return math_ops.maximum(alpha, layer)

    def my_softmax(self, layer, update = True):
        trLayer = tf.transpose(layer)
        print("tf.transpose(layer)- " + str(tf.transpose(layer).shape))
        print("tf.reduce_max(trLayer, reduction_indices = [0])- " + str(tf.reduce_max(trLayer, reduction_indices = [0]).shape))
        print("tf.transpose(trLayer - tf.reduce_max(trLayer, reduction_indices = [0]))- " + str(tf.transpose(trLayer - tf.reduce_max(trLayer, reduction_indices = [0])).shape))
        # print("layer - tf.reduce_max(layer, reduction_indices = [1])- " + str(layer - tf.reduce_max(layer, reduction_indices = [1]).shape))
        print("Done")
        
        expStable = tf.exp(tf.transpose(trLayer - tf.reduce_max(trLayer, reduction_indices = [0])))
        print("Shape of tf.nn.softmax(self.output)- " + str(tf.nn.softmax(self.output).shape))
        
        trExp = tf.transpose(expStable)
        print("tf.transpose(trExp/tf.reduce_sum(trExp, reduction_indices = [0]))- " + str(tf.transpose(trExp/tf.reduce_sum(trExp, reduction_indices = [0])).shape))
        return tf.transpose(trExp/tf.reduce_sum(trExp, reduction_indices = [0]))

    def cross_entropy_loss(self, yPred, y, update = True):
        condition = tf.equal(yPred, 0.)
        yPred_new = tf.add(1e-15, yPred)
        term = tf.reduce_mean(-tf.reduce_sum(y*tf.log(yPred_new)))
        print(term)
        return term

def train():
    train_data, _train_labels = load_data()
    train_labels = np.zeros((len(_train_labels),10))
    for i in xrange(len(_train_labels)):
        train_labels[i][_train_labels[i]] = 1
    
    val_split_len = int(0.7 * len(_train_labels))
    val_data = train_data[val_split_len:]
    val_labels = train_labels[val_split_len:]
    train_data = train_data[:val_split_len]
    train_labels = train_labels[:val_split_len]
    learning_rate = 0.001
    training_epochs = 50
    batch_size = 100
    hidden_dim = 300
    nn = NN(train_data.shape[1],train_labels.shape[1], hidden_dim, learning_rate)
    best_validation_accuracy = 0.0

    # Iteration-number for last improvement to validation accuracy.
    last_improvement = 0

    # Stop optimization if no improvement found in this many iterations.
    patience = 10

    # Start session
    sv = tf.train.Supervisor(graph=nn.graph,
                            logdir='weights/',
                            summary_op=None,
                            save_model_secs=0)

    with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        for epoch in xrange(training_epochs):
            avg_cost = 0
            total_batch = int(len(train_data) / batch_size)
            if sv.should_stop(): 
                break
            for i in xrange(total_batch):
                batch_xs, batch_ys = train_data[(i)*batch_size:(i+1)*batch_size], train_labels[(i)*batch_size:(i+1)*batch_size]
                feed_dict = {nn.X: batch_xs, nn.Y: batch_ys}
                c, _ = sess.run([nn.cost, nn.optimizer], feed_dict=feed_dict)
                avg_cost += c / total_batch
                if i%50:
                    sv.summary_computed(sess, sess.run(nn.merged, feed_dict))
                    gs = sess.run(nn.global_step, feed_dict)
            
            print 'Epoch : ' + str(epoch) + ' Training Loss: ' + str(avg_cost)
            acc = sess.run(nn.accuracy, feed_dict={nn.X: val_data, nn.Y: val_labels})
            print 'Validation Accuracy: ' + str(acc)
            if acc > best_validation_accuracy:
                last_improvement = epoch
                best_validation_accuracy = acc
                sv.saver.save(sess, 'weights' + '/model_gs', global_step=gs)
            if epoch - last_improvement > patience:
                print("Early stopping ...")
                break

def test():
    sess=tf.Session()
    #First let's load meta graph and restore weights
    metas = glob("./weights/*.meta")
    if not metas:
        print 'Please train the model first using the flag --train'
        exit()
    last_mod = 0
    for meta_file in metas:
        t = path.getmtime(meta_file)
        if t > last_mod:
            last_mod_file = meta_file
            last_mod = t
    saver = tf.train.import_meta_graph(last_mod_file)
    saver.restore(sess,tf.train.latest_checkpoint('./weights/'))

    # Access saved Variables directly
    w1 = sess.run('w1:0')
    b1 = sess.run('b1:0')
    w2 = sess.run('w2:0')
    b2 = sess.run('b2:0')
    w3 = sess.run('w3:0')
    b3 = sess.run('b3:0')
    w4 = sess.run('w4:0')
    b4 = sess.run('b4:0')
    # This will print 2, which is the value of bias that we saved


    # Accessing and creating placeholders variables to create feed-dict to feed new data

    test_data, _test_labels = load_data('test')
    test_labels = np.zeros((len(_test_labels),10))
    for i in xrange(len(_test_labels)):
        test_labels[i][_test_labels[i]] = 1

    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("X:0")
    Y = graph.get_tensor_by_name("Y:0")
    feed_dict ={X:test_data, Y:test_labels}

    # Now, access the op that you want to run. 
    accu = graph.get_tensor_by_name("Accuracy:0")

    print sess.run(accu,feed_dict)
    # This will print 60 which is calculated 
    

def layer(num, update=True):
    sess=tf.Session()
    metas = glob("./weights/*.meta")
    if not metas:
        print 'Please train the model first using the flag --train'
        exit()
    last_mod = 0
    for meta_file in metas:
        t = path.getmtime(meta_file)
        if t > last_mod:
            last_mod_file = meta_file
            last_mod = t
    saver = tf.train.import_meta_graph(last_mod_file)
    saver.restore(sess,tf.train.latest_checkpoint('./weights/'))
    
    w1 = sess.run('w1:0')
    b1 = sess.run('b1:0')
    w2 = sess.run('w2:0')
    b2 = sess.run('b2:0')
    w3 = sess.run('w3:0')
    b3 = sess.run('b3:0')
    w4 = sess.run('w4:0')
    b4 = sess.run('b4:0')

    print w1

    train_data, train_labels = load_data()
    test_data, test_labels = load_data("test")
    
    ip = None
    test_new = None
    if num == 1 :
        ip = np.add(np.matmul(train_data, w1),b1)
        test_new  = np.add(np.matmul(test_data, w1),b1)
    elif num == 2:
        ip = np.add(np.matmul(train_data, w1),b1)
        ip = np.add(np.matmul(ip, w2),b2)
        test_new  = np.add(np.matmul(test_data, w1),b1)
        test_new  = np.add(np.matmul(test_new, w2),b2)
        
    elif num == 3:
        ip = np.add(np.matmul(train_data, w1),b1)
        ip = np.add(np.matmul(ip, w2),b2)
        ip = np.add(np.matmul(ip, w3),b3)
        test_new  = np.add(np.matmul(test_data, w1),b1)
        test_new  = np.add(np.matmul(test_new, w2),b2)
        test_new  = np.add(np.matmul(test_new, w3),b3)
        

    LR = LogisticRegression(max_iter=15)
    print "Training ..."
    LR.fit(ip, train_labels)
    print "Testing ..."
    score = LR.score(test_new,test_labels)
    print "Logistic Regression accuracy = " + str(num) + " = " + str(score)
    # pass




def main():
    parser = argparse.ArgumentParser(description='DL Assignment 3')
    parser.add_argument('--train', action='store_true', help='train mode')
    parser.add_argument('--test', action='store_true', help='test mode')
    parser.add_argument('--layer', default=0, help='layers')

    args = parser.parse_args()
    if args.train :
        print "training mode"
        train()
    elif args.test :
        print "test mode"
        test()
    elif int(args.layer) > 0 and int(args.layer) <= 3 :
        print "layer mode"
        layer(int(args.layer))
    else :
        print "default"
        train()


    # training_images, training_labels = load_data()
    # test_images, test_labels = load_data(mode = 'test')

if __name__ == "__main__":
    main()