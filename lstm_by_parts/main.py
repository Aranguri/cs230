from task import LoadedDictTask, DictTask
import sys
sys.path.append('../../')
from util import *

learning_rate = 1e-4
hidden_size = 512
embeddings_size = 300
batch_size = 32
mem_size = 8
debug_steps = 50
mode = '0.1'

task = LoadedDictTask(batch_size * mem_size)

ws_ids = tf.placeholder(tf.int32, (batch_size, mem_size))
ds_ids = tf.placeholder(tf.int32, (batch_size, mem_size, None))
wq = tf.placeholder(tf.int32, (batch_size, mem_size))
dq_ids = tf.placeholder(tf.int32, (batch_size, None))

embeddings_init = tf.placeholder(tf.float32, (task.vocab_size, embeddings_size))
embeddings = tf.get_variable('embeddings', initializer=embeddings_init)
ws = tf.nn.embedding_lookup(embeddings, ws_ids)
ds = tf.nn.embedding_lookup(embeddings, ds_ids)
dq = tf.nn.embedding_lookup(embeddings, dq_ids)

if mode == '-1':
    similarity = tf.einsum('ijkl,ikl->ij', ds, dq)

if mode == '0.1':
    # similarity: multiply. encode: none
    trans_ds = tf.layers.dense(ds, hidden_size, use_bias=False)
    trans_dq = tf.layers.dense(dq, hidden_size, use_bias=False)
    similarity = tf.einsum('ijkl,ikl->ij', trans_ds, trans_dq)

if mode == '0.2':
    # similarity: multiply. encode: none
    ds_h1 = tf.layers.dense(ds, hidden_size, activation=tf.nn.relu)
    ds_h2 = tf.layers.dense(ds_h1, hidden_size, activation=tf.nn.relu)
    trans_ds = tf.layers.dense(ds_h2, hidden_size)
    dq_h1 = tf.layers.dense(dq, hidden_size, activation=tf.nn.relu)
    dq_h2 = tf.layers.dense(dq_h1, hidden_size, activation=tf.nn.relu)
    trans_dq = tf.layers.dense(dq_h2, hidden_size)
    similarity = tf.einsum('ijkl,ikl->ij', trans_ds, trans_dq)

if mode[0] == '1':
    ds_reshaped = tf.reshape(ds, (batch_size * mem_size, -1, embeddings_size))
    defs = tf.concat((ds_reshaped, dq), axis=0)

    if mode == '1.1':
        #1.1: similarity: multiply. encode: rnn
        rnn = tf.contrib.rnn.LSTMCell(hidden_size)
        outputs, _ = tf.nn.dynamic_rnn(rnn, defs, dtype=tf.float32)

        trans_ds, trans_dq = outputs[:-batch_size, -1], outputs[-batch_size:, -1]
        # a transformation here could be useful
        trans_ds = tf.reshape(trans_ds, (batch_size, mem_size, hidden_size))
        similarity = tf.einsum('ijk,ik->ij', trans_ds, trans_dq)

    if mode == '1.2':
        #1.2: similarity: multiply. encode: birnn + softmax
        rnn_fw = tf.contrib.rnn.LSTMCell(hidden_size)
        rnn_bw = tf.contrib.rnn.LSTMCell(hidden_size)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_fw, rnn_bw, defs, dtype=tf.float32)
        trans_ds, trans_dq = tf.split(outputs, [batch_size * mem_size, batch_size], axis=1)

        seq_length = tf.shape(trans_ds)[2]# are both seq_length equal?
        trans_ds = tf.reshape(trans_ds, (2, batch_size, mem_size, seq_length, hidden_size)) #try whether merging this and the next line changes accuracy
        trans_ds = tf.reshape(trans_ds, (batch_size, mem_size, seq_length, 2 * hidden_size))
        trans_dq = tf.reshape(trans_dq, (batch_size, seq_length, 2 * hidden_size)) #todo:change name of trans_ds/dq vars

        def transform(defs):
            defs, probs = tf.split(defs, [2 * hidden_size - 1, 1], axis=-1)
            probs = tf.nn.softmax(probs, axis=-2)
            combined = tf.reduce_sum(defs * probs, axis=-2)
            #add a layer dense heree
            return combined

        trans_ds = transform(trans_ds)
        trans_dq = transform(trans_dq)
        similarity = tf.einsum('ijk,ik->ij', trans_ds, trans_dq)

# Loss and accuracy
correct_cases = tf.equal(tf.argmax(similarity, 1), tf.argmax(wq, 1))
accuracy = tf.reduce_mean(tf.to_float(correct_cases))
loss = tf.losses.softmax_cross_entropy(wq, similarity)
optimizer = tf.train.AdamOptimizer(learning_rate)
minimize = optimizer.minimize(loss)

def next_batch_wrapper(train=True):
    defs1, defs2, words, (m1, m2, w) = task.next_batch() if train else task.dev_batch()

    defs1 = defs1.reshape(batch_size, mem_size, -1)
    defs2 = defs2.reshape(batch_size, mem_size, -1)
    words = np.reshape(words, (batch_size, mem_size))
    m1 = np.array(m1).reshape(batch_size, mem_size)
    m2 = np.array(m2).reshape(batch_size, mem_size)
    query_ixs = np.random.randint(mem_size, size=(batch_size,))
    query_words = one_hot(query_ixs, mem_size)
    query_defs = defs2[range(batch_size), query_ixs]
    query_m2 = m2[range(batch_size), query_ixs]
    query_w = np.array(w)[query_ixs]
    return {ws_ids: words, ds_ids: defs1, wq: query_words, dq_ids: query_defs}, m1, query_m2, query_w

with tf.Session() as sess:
    glove_embeddings = task.glove_embeddings()
    sess.run(tf.global_variables_initializer(), feed_dict={embeddings_init: glove_embeddings})
    tr_acc, dev_loss, dev_acc, sim = {}, {}, {}, {}

    for i in itertools.count():
        feed_dict = next_batch_wrapper()[0]
        tr_acc[i], similarity_ = sess.run([accuracy, similarity], feed_dict)

        '''
        if i % 25 == 0:
            ds_ = dict_task.embed_sentence('I ate a banana')
            ds_ = np.tile(ds_, (4, 8, 1, 1))
            dq_ = dict_task.embed_sentence('I ate a banana')
            dq_ = np.tile(dq_, (4, 1, 1))
            similarity_ = sess.run([similarity], feed_dict={ds: ds_, dq: dq_})
            sim[i//25] = similarity_[0][0][0]
            smooth_plot(sim, 25)
        '''

        if i % debug_steps == 0:
            temp_loss, temp_acc = np.zeros((2, task.dev_batches))
            for j in range(task.dev_batches):
                feed_dict, m1, qm2, qw = next_batch_wrapper(train=False)
                temp_loss[j], temp_acc[j], similarity_ = sess.run([loss, accuracy, similarity], feed_dict)
            # print(similarity_[0])
            # print(m1[0])
            # print(qm2[0])
            # print(qw[0])
            # print()


            dev_loss[i//debug_steps] = temp_loss.mean()
            dev_acc[i//debug_steps] = temp_acc.mean()
            # similarity_ = np.expand_dims(np.argmax(similarity_, 1), 1)
            # correct = np.argmax(similarity_, 1) == feed_dict[wq]
            # print(np.mean(list(tr_acc.values())))
            debug(i, tr_acc, dev_acc, debug_steps)
            # ps(correct)
            '''
            for batch_d1, batch_d2, batch_c in zip(defs1, defs2, correct):
                for d1, d2, c in zip(batch_d1, batch_d2, batch_c):
                    print(''.join(['_']*100))
                    print(d1)
                    print(d2)
                    print(c)
            '''
        # if i % debug_steps * 100 == 0:



'''
# probs = tf.nn.softmax(similarity)
# wq_guess = tf.einsum('ij,ijl->il', probs, ws)
#do we want to transform wq_guess?'''
