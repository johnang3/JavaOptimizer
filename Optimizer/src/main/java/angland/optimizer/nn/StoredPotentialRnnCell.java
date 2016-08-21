package angland.optimizer.nn;

import java.util.Map;

import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.matrix.Matrix;
import angland.optimizer.var.scalar.Scalar;

public class StoredPotentialRnnCell implements RnnCell<String> {

  private final int size;
  private final Matrix<String> updateWeights;
  private final FeedForwardLayer<String> selector;
  private final double gradientClipThreshold;


  public StoredPotentialRnnCell(String varPrefix, int size,
      Map<IndexedKey<String>, Double> context, double gradientClipThreshold, boolean isConstant) {
    super();
    this.size = size;
    this.updateWeights = Matrix.var("adjust_weights", size, 2 * size, context);
    this.selector =
        new FeedForwardLayer<>(2 * size, size, v -> v.tanh().clipGradient(gradientClipThreshold)
            .cache(), varPrefix + "_select_w", varPrefix + "_select_b", context, isConstant);
    this.gradientClipThreshold = gradientClipThreshold;
  }

  @Override
  public RnnStateTuple<String> apply(RnnStateTuple<String> input) {
    Matrix<String> combinedInputs = input.getHiddenState().vCat(input.getExposedState());
    Matrix<String> update =
        updateWeights.streamingTimes(combinedInputs).transform(
            s -> s.tanh().clipGradient(gradientClipThreshold).cache());
    Matrix<String> updatedPotential = input.getHiddenState().plus(update);
    Matrix<String> newCombined = updatedPotential.vCat(input.getExposedState());
    Matrix<String> selection = selector.apply(newCombined);
    Matrix<String> chosen =
        selection.pointwiseMultiply(updatedPotential).transform(
            x -> x.tanh().clipGradient(gradientClipThreshold).cache());
    Matrix<String> lastPotential = updatedPotential.pointwise(chosen, Scalar::minus);
    return new RnnStateTuple<>(lastPotential.transform(Scalar::cache), chosen);
  }

  @Override
  public int getSize() {
    return size;
  }

}
