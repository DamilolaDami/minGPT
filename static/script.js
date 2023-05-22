$(document).ready(function() {
    $("#generate-btn").click(function() {
        var text = $("#input-text").val();

        $.ajax({
            url: "/generate",
            type: "POST",
            dataType: "json",
            contentType: "application/json",
            data: JSON.stringify({ "text": text, "n_emb": $("#n-emb").val(), "n_head": $("#n-head").val() }),
            beforeSend: function() {
                //clear previous output
                $("#output-text").html("");
                // Show loading spinner
                $(".container").append('<div class="loader"></div>');
            },
            success: function(response) {
                var generatedText = response.generated_text;
                generatedText = generatedText.replace(/\n/g, "<br>"); // Replace newline characters with <br> tags
                $("#output-text").html(generatedText);
            },
            error: function(xhr, status, error) {
                console.log("Error: " + error);
                console.log("Status: " + status);
                console.log(xhr);

                // Show error alert
                $("#output-text").before('<div class="alert">An error occurred. Please try again.</div>');
            },
            complete: function() {
                // Remove the loading spinner
                $(".loader").remove();
            }
        });
    });
});