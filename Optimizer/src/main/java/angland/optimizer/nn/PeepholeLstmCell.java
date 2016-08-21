package angland.optimizer.nn;

import angland.optimizer.var.Context;
import angland.optimizer.var.matrix.Matrix;
import angland.optimizer.var.scalar.Scalar;

public class PeepholeLstmCell implements RnnCell<String> {

  private static final Scalar<String> one = Scalar.constant(1);
  private static final Scalar<String> minusOne = Scalar.constant(-1);
  private final FeedForwardLayer<String> retainLayer;
  private final FeedForwardLayer<String> modifyLayer;
  private final FeedForwardLayer<String> selectLayer;
  private final double gradientClipThreshold;
  private final int size;


  public PeepholeLstmCell(String varPrefix, int size, Context<String> context,
      double gradientClipThreshold, boolean constant) {
    this.retainLayer =
        new FeedForwardLayer<>(2 * size, size, v -> v.sigmoid().clipGradient(gradientClipThreshold)
            .cache(2 * size * size), varPrefix + "_retain_w", varPrefix + "_retain_b", context,
            constant);
    this.modifyLayer =
        new FeedForwardLayer<>(2 * size, size, v -> v.tanh().clipGradient(gradientClipThreshold)
            .cache(2 * size * size), varPrefix + "_modify_w", varPrefix + "_modify_b", context,
            constant);
    this.selectLayer =
        new FeedForwardLayer<>(2 * size, size, v -> v.sigmoid().clipGradient(gradientClipThreshold)
            .cache(2 * size * size), varPrefix + "_select_w", varPrefix + "_select_b", context,
            constant);
    this.gradientClipThreshold = gradientClipThreshold;
    this.size = size;
  }



  public double getGradientClipThreshold() {
    return gradientClipThreshold;
  }

  @Override
  public RnnStateTuple<String> apply(RnnStateTuple<String> input) {

    Matrix<String> combinedInputs = input.getHiddenState().vCat(input.getExposedState());

    Matrix<String> retainHidden = retainLayer.apply(combinedInputs);

    Matrix<String> replaceHidden = minusOne.times(retainHidden).transform(s -> s.plus(one));

    Matrix<String> modifier = modifyLayer.apply(combinedInputs);
    Matrix<String> replaceModify = replaceHidden.pointwiseMultiply(modifier);

    Matrix<String> hiddenModified =
        input.getHiddenState().pointwiseMultiply(retainHidden).plus(replaceModify)
            .transform(x -> x.cache(2 * size * size));

    Matrix<String> combinedUpdated = hiddenModified.vCat(input.getExposedState());

    Matrix<String> selector = selectLayer.apply(combinedUpdated);
    Matrix<String> selectedOutput = selector.pointwiseMultiply(hiddenModified);
    return new RnnStateTuple<>(hiddenModified, selectedOutput);
  }

  @Override
  public int getSize() {
    return size;
  }
}
