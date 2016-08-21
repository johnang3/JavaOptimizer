package angland.optimizer.nn;

import angland.optimizer.var.Context;
import angland.optimizer.var.matrix.Matrix;
import angland.optimizer.var.scalar.Scalar;


public class LstmCell<VarKey> implements RnnCell<String> {

  private FeedForwardLayer<String> retain;

  private static final Scalar<String> one = Scalar.constant(1);
  private static final Scalar<String> minusOne = Scalar.constant(1);
  private final int size;
  private FeedForwardLayer<String> modify;
  private FeedForwardLayer<String> select;
  private final double gradientClipThreshold;

  public LstmCell(String varPrefix, int size, Context<String> context,
      double gradientClipThreshold, boolean constant) {
    this.retain =
        new FeedForwardLayer<>(size, size, v -> v.sigmoid().clipGradient(gradientClipThreshold),
            varPrefix + "_retain_w", varPrefix + "_retain_b", context, constant);
    this.modify =
        new FeedForwardLayer<>(size, size, v -> v.tanh().clipGradient(gradientClipThreshold),
            varPrefix + "_modify_w", varPrefix + "_modify_b", context, constant);
    this.select =
        new FeedForwardLayer<>(size, size, v -> v.sigmoid().clipGradient(gradientClipThreshold),
            varPrefix + "_select_w", varPrefix + "_select_b", context, constant);
    this.gradientClipThreshold = gradientClipThreshold;
    this.size = size;
  }



  public double getGradientClipThreshold() {
    return gradientClipThreshold;
  }



  public RnnStateTuple<String> apply(RnnStateTuple<String> input) {

    Matrix<String> retainHidden = retain.apply(input.getExposedState());

    Matrix<String> replaceHidden = minusOne.times(retainHidden).transform(s -> s.plus(one));

    Matrix<String> modifier = modify.apply(input.getExposedState());
    Matrix<String> replaceModify = replaceHidden.pointwiseMultiply(modifier);

    Matrix<String> hiddenModified =
        input.getHiddenState().pointwiseMultiply(retainHidden).plus(replaceModify);


    Matrix<String> selector = select.apply(input.getExposedState());

    Matrix<String> cellOutput = selector.pointwiseMultiply(hiddenModified);

    return new RnnStateTuple<>(hiddenModified.transform(Scalar::cache),
        cellOutput.transform(x -> x.clipGradient(gradientClipThreshold)));
  }

  public int getSize() {
    return size;
  }



}
