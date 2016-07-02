package angland.optimizer.nn;

import java.util.function.Function;
import java.util.stream.Stream;

import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.MatrixExpression;
import angland.optimizer.var.ScalarValue;

public class FeedForwardLayer<VarKey> {

  private final int inputSize;
  private final int outputSize;
  private final MatrixExpression<VarKey> weights;
  private final MatrixExpression<VarKey> biases;
  private final Function<ScalarValue<VarKey>, ScalarValue<VarKey>> transformation;
  private final VarKey weightKey;
  private final VarKey biasKey;

  public FeedForwardLayer(int inputSize, int outputSize,
      Function<ScalarValue<VarKey>, ScalarValue<VarKey>> transformation, VarKey weightKey,
      VarKey biasKey) {
    super();
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.weights = MatrixExpression.variable(weightKey, this.outputSize, this.inputSize);
    this.biases = MatrixExpression.variable(biasKey, this.outputSize, 1);
    this.transformation = transformation;
    this.weightKey = weightKey;
    this.biasKey = biasKey;
  }

  public MatrixExpression<VarKey> apply(MatrixExpression<VarKey> input) {
    return weights.times(input).plus(biases).transform(transformation);
  }

  public int getInputSize() {
    return inputSize;
  }

  public int getOutputSize() {
    return outputSize;
  }

  public MatrixExpression<VarKey> getWeights() {
    return weights;
  }

  public MatrixExpression<VarKey> getBiases() {
    return biases;
  }

  public Stream<IndexedKey<VarKey>> getVarKeys() {
    return Stream.concat(IndexedKey.getAllMatrixKeys(weightKey, this.outputSize, this.inputSize)
        .stream(), IndexedKey.getAllMatrixKeys(this.biasKey, this.outputSize, 1).stream());
  }

}
