package angland.optimizer.nn;

import java.util.function.Function;
import java.util.stream.Stream;

import angland.optimizer.var.Context;
import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.matrix.IMatrixValue;
import angland.optimizer.var.scalar.IScalarValue;

public class FeedForwardLayer<VarKey> {

  private final int inputSize;
  private final int outputSize;
  private final IMatrixValue<VarKey> weights;
  private final IMatrixValue<VarKey> biases;
  private final Function<IScalarValue<VarKey>, IScalarValue<VarKey>> transformation;

  public FeedForwardLayer(int inputSize, int outputSize,
      Function<IScalarValue<VarKey>, IScalarValue<VarKey>> transformation, VarKey weightKey,
      VarKey biasKey, Context<VarKey> context, boolean constant) {
    super();
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.weights =
        IMatrixValue.varOrConst(weightKey, this.outputSize, this.inputSize, context, constant);
    this.biases = IMatrixValue.varOrConst(biasKey, this.outputSize, 1, context, constant);
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
      int inputSize, int outputSize) {
    return Stream.concat(IndexedKey.getAllMatrixKeys(weightKey, outputSize, inputSize).stream(),
        IndexedKey.getAllMatrixKeys(biasKey, outputSize, 1).stream());
  }

}
