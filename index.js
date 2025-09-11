async function klassification() {
    const LEARNING_RATE = 0.001;
    const OPTIMIZER = tf.train.sgd(LEARNING_RATE);
    const INPUTS = [];

    // ТРЕНИРОВКА
    for (let i = 0; i <= 100; i++) {
        INPUTS.push(i);
    }
    const OUTPUT = [];
    INPUTS.forEach((item, index) => {
        if (INPUTS[index] < 50) {
            OUTPUT.push(0);
        } else {
            OUTPUT.push(1);
        }
    });
    const model = tf.sequential();
    model.add(
        tf.layers.dense({
            units: 1,
            activation: 'sigmoid',
            inputShape: 1,
            weights: [tf.tensor2d([3.5], [1, 1]), tf.tensor1d([-200])],
        }),
    );
    model.summary();
    model.compile({ loss: 'meanSquaredError', optimizer: OPTIMIZER });
    const xs = tf.tensor1d(INPUTS);
    const ys = tf.tensor1d(OUTPUT);

    const result = await model.fit(xs, ys, {
        epochs: 400,
        batchSize: 2,
        shuffle: true,
        callbacks: { onEpochEnd: logProgress },
    });
    function logProgress(epoch, logs) {
        console.log('Data for epoch ' + epoch, Math.sqrt(logs.loss));
    }

    for (let i = 0; i < model.getWeights().length; i++) {
        console.log(model.getWeights()[i].dataSync());
    }

    // ПРЕДСКАЗАНИЕ
    const search = [];
    for (let i = 1; i <= 99; i++) {
        search.push(i);
    }
    const predictions = model.predict(tf.tensor1d(search)).dataSync();
    /*     document.getElementById('micro-out-div').innerText = predictions; */
    console.log('Avarage error loss ' + Math.sqrt(result.history.loss[result.history.loss.length - 1]));

    createGraph(INPUTS, OUTPUT, search, predictions);
}

klassification();

// ГРАФИКИ
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
        mode: 'markers',
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
