
//AutoComplete
$("#search").autocomplete({
        source: function(data, cb){
            $.ajax ({
                url:"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords=" + $('#search').val() + "&apikey=JONVXPOKNF9K97BM",
                dataType:'json',
                method: 'GET',
                minLength: 3,
                //data:{id:name},
                success:function(dataapi){
                    var defrelt = dataapi.bestMatches

                    var d = $.map(defrelt, function(item){
                        return{

                            lable: item['2. name'],
                            value: item['1. symbol'] + " - " + item['2. name'] + " ---> " + item['4. region'],

                        }

                    })

                    cb(d)

                 }

            })
        }


    })




