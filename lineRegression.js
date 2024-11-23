import { getRandomInt } from "./lib/getRandomInt.js";

function linearRegression(features, labels, learningRate = 0.1, epochs = 1000) {
  let pricePerRoom = getRandomInt(1, 1000);
  let basePrice = getRandomInt(1, 1000);
  for (let epoch = 0; epoch <= epochs; epoch++) {
    const i = getRandomInt(0, features.length - 1);
    const numRooms = features[i];
    const price = labels[i];
    const newParams = squereTrick(
      basePrice,
      pricePerRoom,
      numRooms,
      price,
      learningRate
    );
    pricePerRoom = newParams.pricePerRoom;
    basePrice = newParams.basePrice;
  }
  return { pricePerRoom, basePrice };
}

// Квадратический подход
function squereTrick(basePrice, pricePerRoom, numRooms, price, learningRate) {
  const predictedPrice = basePrice + pricePerRoom * numRooms;
  basePrice += learningRate * (price - predictedPrice);
  pricePerRoom += learningRate * numRooms * (price - predictedPrice);
  return { pricePerRoom, basePrice };
}

const labels = [150, 200, 250, 300];
const features = [1, 2, 3, 4];

console.log(linearRegression(features, labels));
