const layerDefs = [];

layerDefs.push({ type: 'input', out_sx: 32, out_sy: 32, out_depth: 3 });
layerDefs.push({ type: 'conv', sx: 5, filters: 16, stride: 1, pad: 2, activation: 'relu' });
layerDefs.push({ type: 'pool', sx: 2, stride: 2 });
layerDefs.push({ type: 'conv', sx: 5, filters: 20, stride: 1, pad: 2, activation: 'relu' });
layerDefs.push({ type: 'pool', sx: 2, stride: 2 });
layerDefs.push({ type: 'conv', sx: 5, filters: 20, stride: 1, pad: 2, activation: 'relu' });
layerDefs.push({ type: 'pool', sx: 2, stride: 2 });
layerDefs.push({ type: 'softmax', num_classes: 10 });
  
const net = new convnetjs.Net();
net.makeLayers(layerDefs);

const trainer = new convnetjs.SGDTrainer(net, { method: 'adadelta', batch_size: 4, l2_decay: 0.0001 });
console.log(trainer);
