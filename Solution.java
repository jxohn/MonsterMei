import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Created by XJohn on 2017/11/25.
 */
public class Solution {
    public Solution() {

    }

    public List<Pag> findBestSolution(List<Pag> list, int n){
        return null;
    }
    public static List<Pag> sortList(List<Pag> list){
        Pag tempPag;
        for (int i = 0; i < list.size(); i++) {
            for (int j = i+1; j < list.size()-1; j++) {
                if(list.get(i).getEvg() < list.get(j).getEvg()){
                    tempPag = list.get(i);
                    list.set(i, list.get(j)) ;
                    list.set(j, tempPag);
                }
            }
        }
        return list;
    }

    public static void main(String[] args) {
        List<Integer> weightList = new ArrayList<>();
        weightList.add(4);
        weightList.add(7);
        weightList.add(5);
        weightList.add(3);
        List<Integer> valueList = new ArrayList<>();
        valueList.add(40);
        valueList.add(42);
        valueList.add(25);
        valueList.add(12);

        int n = 10;

        List<Pag> list = new ArrayList<>();
        for (int i = 0; i < weightList.size(); i++) {
            Pag pag = new Pag();
            pag.setWeight(weightList.get(i));
            pag.setValue(valueList.get(i));
            pag.setEvg(weightList.get(i)/valueList.get(i));
            list.add(pag);
        }
        List<Pag> sortList = sortList(list);


    }

//    private static List<Map<String, Integer>> bound(){
//
//    }
}
