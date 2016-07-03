package angland.optimizer.nn;

import java.util.stream.Stream;

import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.MatrixExpression;
import angland.optimizer.var.ScalarExpression;
import angland.optimizer.var.ScalarValue;


public class LstmCell {

  private FeedForwardLayer<String> retain;

  private static final ScalarExpression<String> one = ScalarExpression.constant(1);
  private static final ScalarExpression<String> minusOne = ScalarExpression.constant(1);
  private final int size;
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
    this.size = size;
  }

  public LstmStateTupleExpression<String> apply(LstmStateTupleExpression<String> input) {

    MatrixExpression<String> retainHidden = retain.apply(input.getExposedState());

    MatrixExpression<String> replaceHidden = minusOne.times(retainHidden).addToAll(one);

    MatrixExpression<String> modifier = modify.apply(input.getExposedState());
    MatrixExpression<String> replaceModify = replaceHidden.pointwiseMultiply(modifier);

    MatrixExpression<String> hiddenModified =
        input.getHiddenState().pointwiseMultiply(retainHidden).plus(replaceModify);


    MatrixExpression<String> selector = select.apply(input.getExposedState());

    MatrixExpression<String> cellOutput = selector.pointwiseMultiply(hiddenModified);

    return new LstmStateTupleExpression<>(hiddenModified, cellOutput);
  }

  public int getSize() {
    return size;
  }

  public Stream<IndexedKey<String>> getKeys() {
    return Stream.concat(modify.getVarKeys(),
        Stream.concat(select.getVarKeys(), retain.getVarKeys()));
  }


}
