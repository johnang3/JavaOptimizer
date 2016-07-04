package angland.optimizer.nn;

import java.util.Map;
import java.util.function.Function;
import java.util.stream.Stream;

import angland.optimizer.var.IMatrixValue;
import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.ScalarValue;

public class FeedForwardLayer<VarKey> {

  private final int inputSize;
  private final int outputSize;
  private final IMatrixValue<VarKey> weights;
  private final IMatrixValue<VarKey> biases;
  private final Function<ScalarValue<VarKey>, ScalarValue<VarKey>> transformation;

  public FeedForwardLayer(int inputSize, int outputSize,
      Function<ScalarValue<VarKey>, ScalarValue<VarKey>> transformation, VarKey weightKey,
      VarKey biasKey, Map<IndexedKey<VarKey>, Double> context) {
    super();
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.weights = IMatrixValue.var(weightKey, this.outputSize, this.inputSize, context);
    this.biases = IMatrixValue.var(biasKey, this.outputSize, 1, context);
    this.transformation = transformation;

  }

  public IMatrixValue<VarKey> apply(IMatrixValue<VarKey> input) {
    return weights.times(input).plus(biases).transform(transformation);
  }

  public int getInputSize() {
    return inputSize;
  }

  public int getOutputSize() {
    return outputSize;
  }

  public IMatrixValue<VarKey> getWeights() {
    return weights;
  }

  public IMatrixValue<VarKey> getBiases() {
    return biases;
  }

  public static <VarKey> Stream<IndexedKey<VarKey>> getVarKeys(VarKey weightKey, VarKey biasKey,
      int outputSize, int inputSize) {
    return Stream.concat(IndexedKey.getAllMatrixKeys(weightKey, outputSize, inputSize).stream(),
        IndexedKey.getAllMatrixKeys(biasKey, outputSize, 1).stream());
  }

}
