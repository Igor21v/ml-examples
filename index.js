async function leaner() {
    const LEARNING_RATE = 0.01;
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    model.compile({ loss: 'meanSquaredError', optimizer: tf.train.sgd(LEARNING_RATE) });
    // Generate some synthetic data for training. (y = 2x - 1)
    const xs = tf.tensor1d([-1, 0, 1, 2, 3, 4, 5]);
    const ys = tf.tensor1d([-3, -1, 1, 3, 5, 7, 9]);
    await model.fit(xs, ys, { epochs: 300 });
    document.getElementById('micro-out-div').innerText = model.predict(tf.tensor1d([4])).dataSync();
    const output = model.predict(tf.tensor1d([3]));
    output.print();
    await model.save('downloads://my-model');
}

leaner();

function polinom() {
    // Fit a quadratic function by learning the coefficients a, b, c.
    const xs = tf.tensor1d([0, 1, 2, 3]);
    const ys = tf.tensor1d([1.1, 5.9, 16.8, 33.9]);

    const a = tf.scalar(Math.random()).variable();
    const b = tf.scalar(Math.random()).variable();
    const c = tf.scalar(Math.random()).variable();

    // y = a * x^2 + b * x + c.
    const f = (x) => a.mul(x.square()).add(b.mul(x)).add(c);
    const loss = (pred, label) => pred.sub(label).square().mean();

    const learningRate = 0.01;
    const optimizer = tf.train.sgd(learningRate);

    // Train the model.
    for (let i = 0; i < 10; i++) {
        optimizer.minimize(() => loss(f(xs), ys));
    }

    // Make predictions.
    console.log(`a: ${a.dataSync()}, b: ${b.dataSync()}, c: ${c.dataSync()}`);
    const preds = f(xs).dataSync();
    preds.forEach((pred, i) => {
        console.log(`x: ${i}, pred: ${pred}`);
    });
}

/* polinom(); */
