package angland.optimizer.demos;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TokenBiMap {

  private final List<String> idxToToken;
  private final Map<String, Integer> tokenToIdx = new HashMap<>();
  private final int unkIdx;

  public TokenBiMap(List<String> idxToToken, String unkSymbol) {
    this.idxToToken = idxToToken;
    for (int i = 0; i < idxToToken.size(); ++i) {
      tokenToIdx.put(idxToToken.get(i), i);
    }
    this.unkIdx = tokenToIdx.get(unkSymbol);
  }

  public String getToken(int idx) {
    return idxToToken.get(idx);
  }

  public int getIdx(String token) {
    return tokenToIdx.getOrDefault(token.toLowerCase(), unkIdx);
  }

  public int getUnkIdx() {
    return unkIdx;
  }



}
