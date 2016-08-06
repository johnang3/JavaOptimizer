package angland.optimizer.nn;

import java.util.stream.Stream;

import angland.optimizer.var.Context;
import angland.optimizer.var.IndexedKey;

public interface RnnCellTemplate {

  public RnnCell<String> create(Context<String> context);

  public Stream<IndexedKey<String>> getKeys();

  public int getSize();

}
