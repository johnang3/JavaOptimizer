package angland.optimizer.nn;

import angland.optimizer.var.Context;
import angland.optimizer.var.matrix.IMatrixValue;
import angland.optimizer.var.scalar.IScalarValue;

public class StoredPotentialRnnCell implements RnnCell<String> {

  private final int size;
  private final IMatrixValue<String> updateWeights;
  private final FeedForwardLayer<String> selector;
  private final double gradientClipThreshold;


  public StoredPotentialRnnCell(String varPrefix, int size, Context<String> context,
      double gradientClipThreshold, boolean isConstant) {
    super();
    this.size = size;
    this.updateWeights = IMatrixValue.var("adjust_weights", size, 2 * size, context);
    this.selector =
        new FeedForwardLayer<>(2 * size, size, v -> v.tanh().clipGradient(gradientClipThreshold)
            .cache(), varPrefix + "_select_w", varPrefix + "_select_b", context, isConstant);
    this.gradientClipThreshold = gradientClipThreshold;
  }

  @Override
  public RnnStateTuple<String> apply(RnnStateTuple<String> input) {
    IMatrixValue<String> combinedInputs = input.getHiddenState().vCat(input.getExposedState());
    IMatrixValue<String> update =
        updateWeights.streamingTimes(combinedInputs).transform(
            s -> s.tanh().clipGradient(gradientClipThreshold).cache());
    IMatrixValue<String> updatedPotential = input.getHiddenState().plus(update);
    IMatrixValue<String> newCombined = updatedPotential.vCat(input.getExposedState());
    IMatrixValue<String> selection = selector.apply(newCombined);
    IMatrixValue<String> chosen =
        selection.pointwiseMultiply(updatedPotential).transform(
            x -> x.tanh().clipGradient(gradientClipThreshold).cache());
    IMatrixValue<String> lastPotential = updatedPotential.pointwise(chosen, IScalarValue::minus);
    return new RnnStateTuple<>(lastPotential.transform(IScalarValue::cache), chosen);
  }

  @Override
  public int getSize() {
    return size;
  }

}
