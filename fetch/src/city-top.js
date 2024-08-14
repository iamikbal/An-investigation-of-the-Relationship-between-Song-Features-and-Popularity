// import necessary modules
const axios = require('axios')
const fs = require('fs')

// import cities name, weeks dates, authorisation tokens
let cities = require('../data/cities')
let weeks = require('../data/weeks')
let token = require('../data/token')


// function to fetch top songs of a city of a week
async function cityTopFetch(city, week) {
    try {
        // making request to API
        const fetchedData = await axios.get(`https://charts-spotify-com-service.spotify.com/auth/v0/charts/citytoptrack-${city}-weekly/${week}`, {
            headers: {
                authorization: token.chartTojen,
            }
        })
        // saving the data into json file
        await fs.promises.writeFile(`./assets/city/${city}${week}.json`, JSON.stringify(fetchedData.data), 'utf-8');
        console.log(`${city}${week} Saved`)
    } catch (error) {
        console.log(`${city}${week} Error`)
        console.log(error)
    }
}

// looping over all cities and weeks
async function main() {
    for (let i = 0; i < cities.length; i++) {
        for (let j = 0; j < weeks.length; j++) {
            try {
                await cityTopFetch(cities[i].toLowerCase(), weeks[j])
            } catch (error) {
                console.log(error)
            }
        }
    }
}

// start
main()

//test code
// cityTopFetch("hyderabad","2023-02-09")
// cityTopFetch("kolkata","2022-12-22")