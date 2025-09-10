async function leaner() {
    const LEARNING_RATE = 0.01;
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    model.compile({ loss: 'meanSquaredError', optimizer: tf.train.sgd(LEARNING_RATE) });
    const xArr = [-1, 0, 1, 2, 3, 4, 5];
    const yArr = [-3, -1, 1, 3, 5, 7, 9];
    // Generate some synthetic data for training. (y = 2x - 1)
    const xs = tf.tensor1d(xArr);
    const ys = tf.tensor1d(yArr);
    await model.fit(xs, ys, {
        epochs: 300,
    });
    const x1 = -1;
    const y1 = model.predict(tf.tensor1d([x1])).dataSync();
    const x2 = 5;
    const y2 = model.predict(tf.tensor1d([x2])).dataSync();

    createGraph(xArr, yArr, [+x1, +x2], [+y1, +y2]);
    /*  document.getElementById('micro-out-div').innerText = model.predict(tf.tensor1d([4])).dataSync();
    const output = model.predict(tf.tensor1d([3]));
    output.print(); */
}

/* leaner(); */

async function parabola() {
    const LEARNING_RATE = 0.0001;
    const OPTIMIZER = tf.train.sgd(LEARNING_RATE);
    const INPUTS = [];
    for (let i = -20; i <= 21; i++) {
        if (i < 10) {
            INPUTS.push([i, 0]);
        } else {
            INPUTS.push([i, 20]);
        }
    }
    console.log(INPUTS);
    const OUTPUT = [];
    INPUTS.forEach((item, index) => {
        OUTPUT.push(INPUTS[index][0] + INPUTS[index][1]);
    });
    console.log(OUTPUT);
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: 2, activation: 'linear' }));
    model.add(tf.layers.dense({ units: 1 }));
    model.summary();
    model.compile({ loss: 'meanSquaredError', optimizer: OPTIMIZER });
    // Generate some synthetic data for training. (y = 2x - 1)
    const xs = tf.tensor2d(INPUTS);
    console.log(xs);
    const ys = tf.tensor1d(OUTPUT);
    /*     console.log('ДО');
    console.log(xs);
    console.log('После');
    console.log(xs.reshape([2, -1])); */
    // validationSplit: 0.15 - разделение на валидационные и обучающие данные
    // batchSize - размер минипакета (количество обработанных точек перед обновлением весов)
    // shuffle - перемешать данные
    const result = await model.fit(xs, ys, {
        epochs: 100,
        batchSize: 2,
        shuffle: true,
        /*  callbacks: { onEpochEnd: logProgress }, */
    });
    /*     for (let i = 0; i < model.getWeights().length; i++) {
        console.log(model.getWeights()[i].dataSync());
    } */
    const search = [];
    for (let i = -19.5; i <= 20; i++) {
        if (i < 10) {
            search.push([i, 0]);
        } else {
            search.push([i, 20]);
        }
    }
    const predictions = model.predict(tf.tensor2d(search)).dataSync();
    document.getElementById('micro-out-div').innerText = predictions;
    console.log('Avarage error loss ' + Math.sqrt(result.history.loss[result.history.loss.length - 1]));
    console.log('Avarage error loss ' + Math.sqrt(result.history.val_loss));
    function logProgress(epoch, logs) {
        console.log('Data for epoch ' + epoch, Math.sqrt(logs.loss));
    }

    const inpG = INPUTS.map((item) => item[0]);
    const searchG = search.map((item) => item[0]);
    createGraph(inpG, OUTPUT, searchG, predictions);
}

parabola();

function polinom() {
    // Fit a quadratic function by learning the coefficients a, b, c.
    const inputs = [];
    for (let i = -20; i <= 20; i++) {
        inputs.push(i);
    }
    const outputs = [];
    inputs.forEach((x) => {
        outputs.push(3 * x * x + 5 * x + 10);
    });
    const xs = tf.tensor1d(inputs);
    const ys = tf.tensor1d(outputs);

    const a = tf.scalar(Math.random()).variable();
    const b = tf.scalar(Math.random()).variable();
    const c = tf.scalar(Math.random()).variable();

    // y = a * x^2 + b * x + c.
    const f = (x) => a.mul(x.square()).add(b.mul(x)).add(c);
    const loss = (pred, label) => pred.sub(label).square().mean();

    const learningRate = 0.00001;
    const optimizer = tf.train.sgd(learningRate);

    // Train the model.
    for (let i = 0; i < 1000; i++) {
        optimizer.minimize(() => loss(f(xs), ys));
    }

    // Make predictions.
    console.log(`a: ${a.dataSync()}, b: ${b.dataSync()}, c: ${c.dataSync()}`);

    const search = [];
    for (let i = -19.5; i <= 20; i++) {
        search.push(i);
    }
    const searchTensor = tf.tensor1d(search);
    const preds = f(searchTensor).dataSync();
    preds.forEach((pred, i) => {
        console.log(`x: ${i}, pred: ${pred}`, `label: ${outputs[i]}`);
    });
    createGraph(inputs, outputs, search, preds);
}

/* polinom(); */

function createGraph(xInp, yInp, xPred, yPred) {
    var trace1 = {
        x: xInp,
        y: yInp,
        mode: 'markers',
        name: 'Входные данные',
    };

    var trace2 = {
        x: xPred,
        y: yPred,
        mode: 'lines',
        name: 'Предсказание модели',
    };

    var data = [trace1, trace2];

    var layout = {
        title: {
            text: 'Оценка модели',
        },
    };

    Plotly.newPlot('myDiv', data, layout);
}
