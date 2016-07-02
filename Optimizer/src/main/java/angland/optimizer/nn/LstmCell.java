package angland.optimizer.nn;

import angland.optimizer.var.MatrixExpression;
import angland.optimizer.var.ScalarExpression;
import angland.optimizer.var.ScalarValue;


public class LstmCell {

  private FeedForwardLayer<String> retain;

  private static final ScalarExpression<String> one = ScalarExpression.constant(1);
  private static final ScalarExpression<String> minusOne = ScalarExpression.constant(1);
  private int size;
  private FeedForwardLayer<String> modify;
  private FeedForwardLayer<String> select;

  public LstmCell(String varPrefix, int size) {
    this.retain =
        new FeedForwardLayer<>(size, size, ScalarValue::sigmoid, varPrefix + "_retain_w", varPrefix
            + "_retain_b");
    this.modify =
        new FeedForwardLayer<>(size, size, ScalarValue::tanh, varPrefix + "_modify_w", varPrefix
            + "_modify_b");
    this.select =
        new FeedForwardLayer<>(size, size, ScalarValue::sigmoid, varPrefix + "_retain_w", varPrefix
            + "_retain_b");
  }

  public LstmStateTuple<String> apply(LstmStateTuple<String> input) {

    MatrixExpression<String> retainHidden = retain.apply(input.getExposedState());

    MatrixExpression<String> replaceHidden = minusOne.times(retainHidden).addToAll(one);

    MatrixExpression<String> modifier = modify.apply(input.getExposedState());
    MatrixExpression<String> replaceModify = replaceHidden.pointwiseMultiply(modifier);

    MatrixExpression<String> hiddenModified =
        input.getHiddenState().pointwiseMultiply(retainHidden).plus(replaceModify);


    MatrixExpression<String> selector = select.apply(input.getExposedState());

    MatrixExpression<String> cellOutput = selector.pointwiseMultiply(hiddenModified);

    return new LstmStateTuple<>(hiddenModified, cellOutput);
  }

  public int getSize() {
    return size;
  }

}
