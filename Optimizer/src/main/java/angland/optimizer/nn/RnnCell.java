package angland.optimizer.nn;

public interface RnnCell<VarKey> {

  public RnnStateTuple<VarKey> apply(RnnStateTuple<VarKey> input);

  public int getSize();

}
