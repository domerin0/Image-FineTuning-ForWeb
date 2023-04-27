const ort = require("onnxruntime-node");
const ImageLoader = require('./ImageLoader.js');
const ndarray = require("ndarray");
const ops = require("ndarray-ops")
const fs = require("fs")
const Tensor = ort.Tensor;
/**
 * Preprocess raw image data to match Resnet50 requirement.
 */
const labelsMap = [
    "bike",
    "car",
    "cat",
    "dog",
    "flower",
    "horse",
    "human",
]
const imageSize = 224;

function preprocess(data, width, height) {
    const dataFromImage = ndarray(new Float32Array(data), [width, height, 4]);
    const dataProcessed = ndarray(new Float32Array(width * height * 3), [1, 3, height, width]);

    // Normalize 0-255 to (-1)-1
    ops.divseq(dataFromImage, 128.0);
    ops.subseq(dataFromImage, 1.0);

    // Realign imageData from [224*224*4] to the correct dimension [1*3*224*224].
    ops.assign(dataProcessed.pick(0, 0, null, null), dataFromImage.pick(null, null, 2));
    ops.assign(dataProcessed.pick(0, 1, null, null), dataFromImage.pick(null, null, 1));
    ops.assign(dataProcessed.pick(0, 2, null, null), dataFromImage.pick(null, null, 0));

    return dataProcessed.data;
}

async function getInputs(){
    // Load image.
    const imageLoader = new ImageLoader(imageSize, imageSize);
    const imageData = await imageLoader.getImageData('./data/data/human/rider-12.jpg');
    // Preprocess the image data to match input dimension requirement
    const width = imageSize;
    const height = imageSize;
    const preprocessedData = preprocess(imageData.data, width, height);
    
    const tensorB = new Tensor('float32', 
        preprocessedData, 
        [1, 3, imageSize, imageSize]
    );
    return tensorB;
}

function indexOfMax(arr) {
    if (arr.length === 0) {
        return -1;
    }

    var max = arr[0];
    var maxIndex = 0;

    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }

    return maxIndex;
}

async function inference(){
    
    // load the ONNX model file
    const cpuSession = await ort.InferenceSession.create('./resnet18.onnx');

    // generate model input
    const inferenceInputs = await getInputs();
    
    // execute the model
    const test = JSON.parse(fs.readFileSync("./test.json"))
    test.data = new Float32Array(Object.values(test.data));
    const output = await cpuSession.run({"input": test});
    console.log(cpuSession.session);
    console.log(labelsMap[indexOfMax(output.output.data)]);
}

inference();