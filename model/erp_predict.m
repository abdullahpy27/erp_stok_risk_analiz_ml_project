function result = erp_predict(stock_quantity, ...
                              weekly_sales, ...
                              supplier_delay_days, ...
                              price, ...
                              last_sale_days, ...
                              seasonality)

persistent model mu sigma

if isempty(model)

    s = load("best_model.mat");

    model = s.rfModel;
    mu = s.mu;
    sigma = s.sigma;

end

if weekly_sales <= 5
    sales_speed_category = 1;
elseif weekly_sales <= 15
    sales_speed_category = 2;
else
    sales_speed_category = 3;
end

X = [
    stock_quantity ...
    weekly_sales ...
    supplier_delay_days ...
    price ...
    last_sale_days ...
    seasonality ...
    sales_speed_category
];

sigma(sigma == 0) = 1;

X = (X - mu) ./ sigma;

prediction = predict(model, X);

result = char(prediction);

end
