package angland.optimizer.nn;

import java.util.function.Function;
import java.util.stream.Stream;

import angland.optimizer.var.Context;
import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.matrix.Matrix;
import angland.optimizer.var.scalar.Scalar;

public class FeedForwardLayer<VarKey> {

  private final int inputSize;
  private final int outputSize;
  private final Matrix<VarKey> weights;
  private final Matrix<VarKey> biases;
  private final Function<Scalar<VarKey>, Scalar<VarKey>> transformation;

  public FeedForwardLayer(int inputSize, int outputSize,
      Function<Scalar<VarKey>, Scalar<VarKey>> transformation, VarKey weightKey,
      VarKey biasKey, Context<VarKey> context, boolean constant) {
    super();
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.weights =
        Matrix.varOrConst(weightKey, this.outputSize, this.inputSize, context, constant);
    this.biases = Matrix.varOrConst(biasKey, this.outputSize, 1, context, constant);
    this.transformation = transformation;

  }

  public Matrix<VarKey> apply(Matrix<VarKey> input) {
    // IScalarValue<VarKey> biasMultiplier = IScalarValue.constant(inputSize);
    return weights.times(input).plus(biases).transform(transformation);
  }

  public int getInputSize() {
    return inputSize;
  }

  public int getOutputSize() {
    return outputSize;
  }

  public Matrix<VarKey> getWeights() {
    return weights;
  }

  public Matrix<VarKey> getBiases() {
    return biases;
  }

  public static <VarKey> Stream<IndexedKey<VarKey>> getVarKeys(VarKey weightKey, VarKey biasKey,
      int inputSize, int outputSize) {
    return Stream.concat(IndexedKey.getAllMatrixKeys(weightKey, outputSize, inputSize).stream(),
        IndexedKey.getAllMatrixKeys(biasKey, outputSize, 1).stream());
  }

}
