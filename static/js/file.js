function readURL(input) {
    if (input.files && input.files[0]) {
  
      var reader = new FileReader();
  
      reader.onload = function(e) {
        $('.image-upload-wrap').hide();
  
        // $('.file-upload-image').attr('src', e.target.result);
        // $('.file-upload-content').show();
  
        $('.image-title').html(input.files[0].name);

      };
  
     reader.readAsDataURL(input.files[0]);
  
    } else {
      removeUpload();
    }
  }
  
  function removeUpload() {
    $('.file-upload-input').replaceWith($('.file-upload-input').clone());
    $('.file-upload-content').hide();
    $('.image-upload-wrap').show();
  }
  $('.image-upload-wrap').bind('dragover', function () {
          $('.image-upload-wrap').addClass('image-dropping');
      });
      $('.image-upload-wrap').bind('dragleave', function () {
          $('.image-upload-wrap').removeClass('image-dropping');
  });
  function processing(){
      $("#loaded-process").css("display","block");
      $("#process-ready").css("display","none");
  }

  function read(name){
    $.ajax({
        type: "POST",
        url: "http://127.0.0.1:5000/voice",
        data: { param: name}
      }).done(function( o ) {
         // do something
      });
  }


  function showSpeech(){
      $("#speech").css("display","block")
  }