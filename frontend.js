$(document).ready(function () {
  $('#searchBar').on('input', function () {
    var searchText = $(this).val();

    $.ajax({
      type: 'POST',
      url: 'http://127.0.0.1:5000/get_recommendations',
      data: JSON.stringify({ searchText: searchText }),
      contentType: 'application/json',
      success: function (response) {
        var recommendationsDiv = $('#recommendations');
        recommendationsDiv.empty();

        if (Array.isArray(response) && response.length) {
          var list = $('<ul>');
          response.forEach(function (movie) {
            list.append($('<li>').text(movie));
          });
          recommendationsDiv.append(list);
        } else {
          recommendationsDiv.text('No recommendations found.');
        }
      },
      error: function (xhr, status, error) {
        console.log('Error:', status, error);
        $('#recommendations').text('Error fetching recommendations.');
      }
    });
  });
});
