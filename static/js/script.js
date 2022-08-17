$(document).ready(function(){

    $(":submit").click(function(e){
        e.preventDefault();
        value = $(this).val()
        $('#model_type').val(value);
        $.ajax({
            type: 'POST',
            url: "/predict",
            data: $('#predictionForm').serialize()
          }).done(function(data){
            console.log(data)
            $('#resultModal').text('')
            $('#resultModal').text(data)
            $('#predictionModal').modal('show')
          });
    });

    $(".close-modal").click(function(){
      $('#predictionModal').modal('hide')
    })

    
})